"""GPT-Hansard-11M inference service (stdlib-only HTTP + torch).

Downloads the checkpoint from the public HF repo at boot, serves:
  GET  /health
  POST /api/generate  {"prompt": str, "temperature": float, "max_new": int}
CORS open so nathanielarfin.com can embed the demo.
"""
import gzip
import json
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from home import PAGE

# Railway caps CPU via cgroups while torch sees every host core — oversubscription
# thrashes. Pin threads low; this alone is often a 5-10x win under quota.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

HF_URL = ("https://huggingface.co/NathanielArfin/gpt-hansard-11m/"
          "resolve/main/gpt-11m-sft-en-v3.pt")
CKPT_PATH = os.environ.get("CKPT_PATH", "/app/gpt-11m-sft-en-v3.pt")
PORT = int(os.environ.get("PORT", 8787))

# ---------------------------------------------------------------- model ----
CAPTURE = None        # when a list, every head stores its attention matrix (T×T)
HEADS = None          # set at boot (heads per layer)


class Head(nn.Module):
    def __init__(self, n_embd, hs, li, hi):
        super().__init__()
        self.key = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
        self.li, self.hi = li, hi

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        att = att.masked_fill(
            torch.tril(torch.ones(x.shape[-2], x.shape[-2])) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if CAPTURE is not None:
            CAPTURE[self.li * HEADS + self.hi] = att[0].detach().to(torch.float32)
        return att @ v


class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head, li):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs, li, hi) for hi in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, li):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHead(n_embd, n_head, li)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block = cfg["block"]
        self.tok_emb = nn.Embedding(cfg["V"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block"], cfg["n_embd"])
        self.blocks = nn.Sequential(*[Block(cfg["n_embd"], cfg["n_head"], li)
                                      for li in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["V"])
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        T = idx.shape[-1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
        return self.head(self.ln_f(self.blocks(x)))


def boot():
    if not os.path.exists(CKPT_PATH):
        print(f"downloading checkpoint from {HF_URL} ...", flush=True)
        req_ok = False
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(HF_URL, CKPT_PATH)
                req_ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"download attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(5)
        if not req_ok:
            raise SystemExit("could not obtain checkpoint")
    ck = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ck["config"]
    global HEADS
    HEADS = cfg["n_head"]
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    try:  # dynamic int8: ~2-4x CPU speedup on Linear-heavy inference
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        print("dynamic int8 quantization applied", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"quantization skipped: {exc}", flush=True)
    mm = {(a, b): i for a, b, i in ck["merges"]}
    tb = {i: bytes([i]) for i in range(256)}
    for a, b, i in ck["merges"]:
        tb[i] = tb[a] + tb[b]
    print(f"model loaded: {cfg} · {n_params:,} params · dynamic int8", flush=True)
    return model, mm, tb


MODEL, MM, TB = boot()
LOCK = threading.Lock()


def encode_bpe(text):
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    for (a, b), idx in MM.items():
        hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if hit.size == 0:
            continue
        if a == b:
            keep, last = [], -2
            for i in hit:
                if i > last + 1:
                    keep.append(i)
                    last = i
            keep = np.array(keep)
        else:
            keep = hit
        ids[keep] = idx
        ids = np.delete(ids, keep + 1)
    return ids


def decode_bpe(ids):
    return b"".join(TB[int(i)] for i in ids).decode("utf-8", errors="replace")


@torch.no_grad()
def generate(prompt, temperature=0.6, max_new=120, rep=1.15, top_k=40, extend=96):
    """Generate, then keep decoding past max_new in a short capped window so the
    answer ends at a completed sentence, not a dangling clause.

    Measured (eos_probe, 12 seeds): the model never emits its own \x00 separator
    or a next-Q scaffold — "conclusion" wasn't learned — so the budget clip was
    leaving the UI mid-sentence. Extending until terminal punctuation (with a
    hard cap), then falling back to cutting at the last completed sentence,
    makes the output read as concluded.
    """
    with LOCK:
        idx = torch.from_numpy(encode_bpe(prompt))[None, :]
        budget = max(1, min(int(max_new), 200))
        recent = idx[0].tolist()
        out = []
        for _ in range(budget):
            logits = MODEL(idx[:, -MODEL.block:])[:, -1, :].clone()
            logits[0, 0] = float("-inf")      # never emit the trained pair separator (\x00)
            if rep and rep > 1.0:             # CTRL-style penalty over the recent window
                for t in set(recent[-64:]):
                    lg = logits[0, t].item()
                    if lg > 0:
                        logits[0, t] = lg - rep * lg
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits / max(0.05, temperature), dim=-1), 1))
            recent.append(nxt)
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        text = decode_bpe(out).replace("\x00", "").strip()
        cut = text.find("\nQ:")               # stop cleanly if it scaffolds the next pair
        if cut != -1:
            text = text[:cut]
        # a spent token budget lands mid-sentence — extend to the next boundary
        if not re.search(r"[.!?…][\"””']?\s*$", text) and not text.endswith(("…",)):
            for _ in range(max(0, int(extend))):
                logits = MODEL(idx[:, -MODEL.block:])[:, -1, :].clone()
                logits[0, 0] = float("-inf")
                if rep and rep > 1.0:
                    for t in set(recent[-64:]):
                        lg = logits[0, t].item()
                        if lg > 0:
                            logits[0, t] = lg - rep * lg
                if top_k and top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                nxt = int(torch.multinomial(F.softmax(logits / max(0.05, temperature), dim=-1), 1))
                recent.append(nxt)
                out.append(nxt)
                idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
                text = decode_bpe(out).replace("\x00", "").strip()
                if re.search(r"[.!?…][\"””']?\s*$", text) or "\nQ:" in text:
                    break
            cut = text.find("\nQ:")
            if cut != -1:
                text = text[:cut]
        # extension cap exhausted mid-clause — fall back to the last completed
        # sentence and mark the trailing-off honestly.
        if text and not re.search(r"[.!?…][\"””']?\s*$", text):
            ends = [m.end() for m in re.finditer(r"[.!?…](?=\s+[A-Z“\"])", text)]
            if ends and ends[-1] >= len(text) * 0.35:
                text = text[:ends[-1]].rstrip() + " …"
        return text


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]      # cache-buster queries must still serve the page
        if path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/health"):
            body = json.dumps({"status": "ok", "model": "gpt-hansard-11m"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _respond_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if "gzip" in (self.headers.get("Accept-Encoding") or ""):
            body = gzip.compress(body)
            self.send_header("Content-Encoding", "gzip")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _attention(self):
        """Attention maps for a prompt: {tokens, layers, heads, attn[l][h][i][j]}.

        attn[i][j] = how much query token i attends to key token j (j ≤ i,
        causal). Educational lens — not an explanation.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            p = str(req.get("prompt", "")).strip()
            if not p:
                raise ValueError("empty prompt")
            wrapped = p if req.get("raw") else ("Q: " + p + "\nA:")
            all_ids = encode_bpe(wrapped)
            ids = all_ids[:96]
            with LOCK:
                global CAPTURE
                CAPTURE = [None] * 6 * HEADS
                with torch.no_grad():
                    MODEL(torch.from_numpy(ids.astype("int64"))[None, :])
                caps, CAPTURE = CAPTURE, None
            T = len(ids)
            attn = [[[round(float(v), 3) for v in caps[l * HEADS + h][t].tolist()]
                     for t in range(T)] for l in range(6) for h in range(HEADS)]
            toks = [TB[int(i)].decode("utf-8", errors="replace") for i in ids.tolist()]
            self._respond_json({"tokens": toks, "layers": 6, "heads": HEADS,
                                "attn": attn, "truncated": len(all_ids) > 96})
        except Exception as exc:  # noqa: BLE001
            self._respond_json({"error": str(exc)}, status=500)

    def do_POST(self):
        if self.path.startswith("/api/attention"):
            self._attention()
            return
        if not self.path.startswith("/api/generate"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            text = generate(req.get("prompt", ""),
                            float(req.get("temperature", 0.6)),
                            int(req.get("max_new", 140)),
                            float(req.get("repetition_penalty", 1.15)))
            body = json.dumps({"text": text}).encode()
            self.send_response(200)
        except Exception as exc:  # noqa: BLE001 — report, keep the server alive
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet
        pass


if __name__ == "__main__":
    print(f"hansard-model listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
