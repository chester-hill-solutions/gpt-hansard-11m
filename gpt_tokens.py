# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5", "numpy", "huggingface_hub"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""
GPT-0 Academy — CAPSTONE: a token-level GPT on Canadian Hansard.

Pipeline: merges.json (1,024 BPE merges, vocab 1,280 — era-sampled on the
2006→now corpus, published with it on Hugging Face) → this script
numpy-encodes the corpus (cached to tokens.bin), trains the scaled model
with a cosine LR schedule, then (optionally, --sft) fine-tunes on the
packed Q/A stream with the SFT lessons baked in:
  - prompt masking (labels -100)      - packing (no padding)
  - model.eval() before generation    - checkpoint carries its tokenizer

Presets:
  default      : n_layer=6 n_head=6 n_embd=384 block=512 (~10.7M params) — Colab T4
  --preset cpu : n_layer=4 n_head=4 n_embd=128 block=256 (~1M params)   — home CPU

Run:
  uv run gpt_tokens.py --preset cpu --steps 2000          # home machine
  uv run gpt_tokens.py --sft                              # after pretraining
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config --
CORPUS_URL = (f"https://huggingface.co/datasets/NathanielArfin/canadian-hansard-2006-now/"
              f"resolve/main/hansard-2006-now.txt")
CORPUS_FILE = "hansard-2006-now.txt"
MERGES_FILE = "merges.json"          # from the era-sampled BPE training run
TOK_CACHE = "tokens.bin"             # 258M-token uint16 stream (HF: canadian-hansard-2006-now)
HF_REPO = "NathanielArfin/canadian-hansard-2006-now"


def hf_fetch(fname: str) -> None:
    """Pull a missing artifact (tokens.bin / merges.json) from the HF dataset repo."""
    print(f"{fname} not found — pulling from huggingface.co/{HF_REPO} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "huggingface_hub"])
        if r.returncode != 0:
            raise SystemExit("huggingface_hub unavailable: install it with "
                             "`pip install huggingface_hub` and retry")
        from huggingface_hub import hf_hub_download
    cached = hf_hub_download(HF_REPO, fname, repo_type="dataset")
    shutil.copyfile(cached, fname)
    print(f"  -> {fname} ({os.path.getsize(fname):,} bytes)")
SLICE = 10_000_000                   # corpus chars → ~3.9M tokens
VAL_FRAC = 0.01

ap = argparse.ArgumentParser()
ap.add_argument("--preset", choices=["default", "cpu", "cpu-big"], default="default")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--batch", type=int, default=None, help="override preset batch size")
ap.add_argument("--sft", action="store_true", help="SFT stage after pretraining")
ap.add_argument("--sft-steps", type=int, default=120)
ap.add_argument("--sft-only", action="store_true",
                help="load --ckpt and run ONLY the SFT stage (no pretraining)")
ap.add_argument("--ckpt", default="gpt_tokens.pt", help="checkpoint path")
ap.add_argument("--sft-data", default="sft/qp-pairs.jsonl",
                help="mined Q/A pairs (jsonl); falls back to built-in pairs")
ap.add_argument("--max-encode-chars", type=int, default=SLICE)
ap.add_argument("--out", default=None,
                help="SFT checkpoint output path (default: gpt_tokens_sft.pt)")
args = ap.parse_args()

if args.preset == "cpu":
    N_LAYER, N_HEAD, N_EMBD, BLOCK, BATCH = 4, 4, 128, 256, 16
    STEPS = args.steps or 2000
    MAX_LR = 6e-4
elif args.preset == "cpu-big":
    N_LAYER, N_HEAD, N_EMBD, BLOCK, BATCH = 6, 6, 192, 256, 16   # ~3.0M params
    STEPS = args.steps or 20000
    MAX_LR = 6e-4
else:
    N_LAYER, N_HEAD, N_EMBD, BLOCK, BATCH = 6, 6, 384, 512, 12
    STEPS = args.steps or 10000
    MAX_LR = 6e-4
if args.batch:
    BATCH = args.batch
MIN_LR = MAX_LR * 0.1
WARMUP = 200
WEIGHT_DECAY = 0.1
DROP_OUT = 0.1
EVAL_EVERY = 250
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ------------------------------------------------------------- tokenizer ----
if not os.path.exists(MERGES_FILE):
    hf_fetch(MERGES_FILE)
raw_merges = json.load(open(MERGES_FILE))            # [[a, b, idx], ...] in order
merges = {(a, b): idx for a, b, idx in raw_merges}
tok_bytes = {i: bytes([i]) for i in range(256)}
for (a, b), idx in merges.items():
    tok_bytes[idx] = tok_bytes[a] + tok_bytes[b]
V = 256 + len(merges)
itos = {i: tok_bytes[i].decode("utf-8", errors="replace") for i in range(V)}
print(f"tokenizer: {V} vocab · {len(merges)} merges")

def encode_bpe(text: str) -> np.ndarray:
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in merges.items():
        hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if hit.size == 0:
            continue
        if a == b:
            keep, last = [], -2                      # 'aa' pairs can overlap ('aaa')
            for i in hit:                            # — greedy left-to-right needed
                if i > last + 1:
                    keep.append(i)
                    last = i
            keep = np.array(keep)
        else:
            keep = hit                               # a≠b ⇒ matches can't touch:
        ids[keep] = idx                              # already non-overlapping
        ids = np.delete(ids, keep + 1)
    return ids.astype(np.uint16)

def decode_bpe(ids) -> str:
    return b"".join(tok_bytes[int(i)] for i in ids).decode("utf-8", errors="replace")

# ------------------------------------------------------------------- data ----
if not os.path.exists(CORPUS_FILE):
    for cand in ("hansard-bilingual.txt", "hansard-2006-now.txt", "hansard-en.txt", "input.txt"):
        if os.path.exists(cand):
            CORPUS_FILE = cand
            break
    else:
        print("fetching corpus ...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_FILE)

if not os.path.exists(TOK_CACHE):
    hf_fetch(TOK_CACHE)
if os.path.exists(TOK_CACHE) and os.environ.get("REBUILD_TOKENS") != "1":
    all_ids = np.fromfile(TOK_CACHE, dtype=np.uint16)
    print(f"token cache: {len(all_ids):,} tokens ({TOK_CACHE})")
else:
    text = open(CORPUS_FILE, encoding="utf-8").read()[:args.max_encode_chars]
    print(f"encoding {len(text):,} chars with {len(merges)} numpy merges ...")
    all_ids = encode_bpe(text)
    all_ids.tofile(TOK_CACHE)
    print(f"cached {len(all_ids):,} tokens -> {TOK_CACHE} "
          f"({len(text) / len(all_ids):.2f} chars/token)")

split = int(len(all_ids) * (1 - VAL_FRAC))
train_ids, val_ids = all_ids[:split], all_ids[split:]

def get_batch(split, batch=BATCH, block=BLOCK):
    src = train_ids if split == "train" else val_ids
    ix = np.random.randint(0, len(src) - block - 1, batch)
    x = torch.stack([torch.from_numpy(src[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(src[i + 1:i + block + 1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ------------------------------------------------------------------ model ----
class Head(nn.Module):
    def __init__(self, n_embd, hs):
        super().__init__()
        self.key = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ v

class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        nn.init.zeros_(self.proj.weight)             # blocks start as no-ops
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHead(n_embd, n_head)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(DROP_OUT))
        nn.init.zeros_(self.mlp[2].weight)           # zero-init residual projections
        nn.init.zeros_(self.mlp[2].bias)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block=BLOCK):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block, n_embd)
        nn.init.normal_(self.tok_emb.weight, std=0.02)   # GPT-2 init — else the tied
        nn.init.normal_(self.pos_emb.weight, std=0.02)   # head makes logits explode
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
        self.head.weight = self.tok_emb.weight        # weight tying
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

model = GPT(V).to(DEVICE)
print(f"model: {sum(p.numel() for p in model.parameters()):,} params · {DEVICE} · "
      f"block {BLOCK} · {STEPS} steps")

def lr_at(step):
    if step < WARMUP:
        return MAX_LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * t))

@torch.no_grad()
def eval_val(n_batches=8):
    model.eval()
    losses = [model(*get_batch("val", batch=BATCH))[1].item() for _ in range(n_batches)]
    model.train()
    return sum(losses) / len(losses)                  # averaged val — the honest kind

# ------------------------------------------------------------- pretraining ---
CKPT = args.ckpt
if args.sft_only:
    ck = torch.load(CKPT, map_location=DEVICE)
    cfg = ck.get("config", {})
    mismatched = [k for k, v in (("V", V), ("n_embd", N_EMBD), ("n_head", N_HEAD),
                                 ("n_layer", N_LAYER), ("block", BLOCK))
                  if cfg.get(k) != v]
    if mismatched:
        raise SystemExit(f"checkpoint {CKPT} config mismatch: {mismatched} "
                         f"(ckpt: {cfg}) — wrong --preset?")
    model.load_state_dict(ck["model"])
    print(f"loaded {CKPT} — sft-only mode, pretraining skipped")
else:
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            print(f"step {step:5d} · train {loss.item():.3f} · val(avg8) {eval_val():.3f} "
                  f"· lr {lr_at(step):.2e}")

    torch.save({"model": model.state_dict(), "merges": raw_merges, "config": {
        "V": V, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "block": BLOCK}},
        CKPT)
    print(f"saved -> {CKPT} (weights + merges + config)")

# ------------------------------------------------------------- generation ----
model.eval()

def gen(prompt: str, max_new=200, temperature=0.8):
    ids = encode_bpe(prompt).astype(np.int64)
    idx = torch.from_numpy(ids)[None, :].to(DEVICE)
    for _ in range(max_new):
        logits = model(idx[:, -BLOCK:])[0][:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return decode_bpe(idx[0].tolist())[len(prompt):]

for p in ["Mr. Speaker, the cost of housing", "Bill C-", "Q: What is Question Period?\nA:"]:
    print(f"\n[prompt] {p!r}")
    print(f"[model ] {p}{gen(p)}")

# ------------------------------------------------------------------- SFT ----
if args.sft:
    print("\n=== SFT stage (packing, prompt-masked, gentle) ===")
    if os.path.exists(args.sft_data):
        # Answer targets carry the responder's Hansard attribution — the model
        # learns to open with the real name/role/riding it saw in pretraining.
        PAIRS = [(r["q"], (f"{r['who_a']}: {r['a']}" if r.get("who_a") else r["a"]))
                 for r in map(json.loads, open(args.sft_data, encoding="utf-8"))
                 if r.get("q") and r.get("a")]
        print(f"SFT data: {len(PAIRS):,} mined pairs from {args.sft_data}")
    else:
        PAIRS = [
            ("What is Hansard?",
             "Hansard is the official transcript of debates in the House of Commons, published in both English and French."),
            ("What is Question Period?",
             "Question Period is the daily time when members ask ministers questions about the government's work."),
            ("Who is the Speaker?",
             "The Speaker is the member who presides over the House and enforces the rules of debate."),
            ("What is a bill?",
             "A bill is a proposed law. It is debated, amended and voted on before it can become law."),
            ("What is royal assent?",
             "Royal assent is the final step at which a bill becomes law, granted by the Governor General."),
            ("What is a committee?",
             "A committee is a small group of members that studies bills and reports to the House."),
            ("What is prorogation?",
             "Prorogation ends a session of Parliament. Bills not passed die on the order paper."),
            ("What is a riding?",
             "A riding is a federal electoral district, represented by one member of Parliament."),
        ]
        print(f"SFT data: built-in {len(PAIRS)} pairs ({args.sft_data} not found)")

    STREAM_CACHE = "sft-stream.bin"
    if os.path.exists(STREAM_CACHE) and os.environ.get("REBUILD_SFT") != "1":
        stream_np = np.fromfile(STREAM_CACHE, dtype=np.uint16).astype(np.int64)
        labels_np = np.fromfile(STREAM_CACHE + ".labels", dtype=np.int32).astype(np.int64)
        print(f"SFT stream: {len(stream_np):,} tokens (cached)")
    else:
        # One sentinel-packed encode: \x00 never occurs in Hansard and no merge
        # contains byte 0, so every \x00 becomes token id 0 and marks boundaries.
        # Pair format: \x00 + "Q: {q}\nA:" + \x00 + " {a}\n" → the segments
        # strictly alternate [prompt, answer] between zeros.
        big = "".join(f"\x00Q: {q}\nA:\x00 {a}\n" for q, a in PAIRS)
        ids = encode_bpe(big).astype(np.int64)
        z = np.flatnonzero(ids == 0)
        bounds = np.concatenate((z[1:], [len(ids)]))  # each segment ends at the NEXT zero
        segs = [ids[b + 1:e] for b, e in zip(z, bounds)]
        segs = [s for s in segs if len(s) > 0]
        if len(segs) % 2 != 0:
            raise SystemExit(f"sentinel packing produced {len(segs)} segments "
                             f"(expected even) — SFT data malformed?")
        stream_np = np.concatenate(segs)
        labels_np = np.concatenate([
            np.full(len(s), -100, dtype=np.int64) if i % 2 == 0 else s
            for i, s in enumerate(segs)])
        stream_np.astype(np.uint16).tofile(STREAM_CACHE)
        labels_np.astype(np.int32).tofile(STREAM_CACHE + ".labels")
        print(f"SFT stream: {len(stream_np):,} tokens (packed, {len(PAIRS):,} pairs, "
              f"prompts masked)")

    stream_t = torch.from_numpy(stream_np)
    labels_t = torch.from_numpy(labels_np)
    N = len(stream_t)

    sft_opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WEIGHT_DECAY)
    model.train()
    for step in range(args.sft_steps):
        ix = torch.randint(max(1, N - BLOCK), (BATCH,))
        xb = stream_t[ix[:, None] + torch.arange(BLOCK)].to(DEVICE)
        yb = labels_t[ix[:, None] + 1 + torch.arange(BLOCK)].to(DEVICE)  # shifted: predict NEXT token
        if (yb != -100).sum() == 0:      # window contains no answer tokens — skip
            continue
        _, loss = model(xb, yb)
        sft_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_opt.step()
        if step % 30 == 0 or step == args.sft_steps - 1:
            print(f"sft step {step:4d} · loss {loss.item():.3f}")

    out_path = args.out or "gpt_tokens_sft.pt"
    torch.save({"model": model.state_dict(), "merges": raw_merges, "config": {
        "V": V, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "block": BLOCK},
        "sft": True}, out_path)
    print(f"saved -> {out_path}")
    model.eval()
    for q in ["What is Hansard?", "What is a filibuster?",
              "Qu'est-ce que la prorogation ?", "Qu'est-ce que le Hansard ?"]:
        print(f"\nQ: {q}\nA: {gen('Q: ' + q + chr(10) + 'A:', max_new=120, temperature=0.7)}")
