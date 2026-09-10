# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5", "numpy"]
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
GPT-0 Academy — eval suite v1.

Judges any gpt_tokens-family checkpoint (weights + merges + config):
  1. held-out perplexity + bits/char on the last 1% of tokens.bin
  2. held-out QP Q/A generations (sft/qp-heldout.jsonl — never trained on)
  3. register metrics: speaker-form rate in the first 160 chars, answer length
  4. repetition: distinct-2 over pooled generations

  uv run eval_hansard.py --ckpt gpt-cpu-small-sft.pt --n-qa 6
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n-qa", type=int, default=6)
ap.add_argument("--gen-tokens", type=int, default=90)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--held-out", default="sft/qp-heldout.jsonl")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
np.random.seed(1337)   # same held-out Q draws across runs for fair A/B

# --------------------------------------------------- checkpoint + tokenizer --
ck = torch.load(args.ckpt, map_location=DEVICE)
cfg = ck["config"]
V = cfg["V"]
merges_list = ck["merges"]
mm = {(a, b): ix for a, b, ix in merges_list}
tok_bytes = {i: bytes([i]) for i in range(256)}
for a, b, ix in merges_list:
    tok_bytes[ix] = tok_bytes[a] + tok_bytes[b]
BLOCK = cfg["block"]


def encode_bpe(text: str) -> np.ndarray:
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in mm.items():
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
    return ids.astype(np.uint16)


def decode_bpe(ids) -> str:
    return b"".join(tok_bytes[int(i)] for i in ids).decode("utf-8", errors="replace")


# ------------------------------------------------------------------- model ---
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
        att = att.masked_fill(
            torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ v


class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHead(n_embd, n_head)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(0.1))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, n_embd, n_head, n_layer, block):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


model = GPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"], BLOCK).to(DEVICE)
model.load_state_dict(ck["model"])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"eval: {args.ckpt} · {n_params:,} params · vocab {V} · block {BLOCK} · {DEVICE}")

# --------------------------------------------------- 1. held-out ppl / bpc ---
if os.path.exists("tokens.bin"):
    all_ids = np.fromfile("tokens.bin", dtype=np.uint16)
    val = all_ids[int(len(all_ids) * 0.99):]          # last 1% — never trained
    cpt = 3.12
    if os.path.exists("tokens-info.json"):
        cpt = json.load(open("tokens-info.json")).get("chars_per_token", 3.12)
    losses = []
    with torch.no_grad():
        for _ in range(16):
            i = np.random.randint(0, len(val) - BLOCK - 1)
            x = torch.from_numpy(val[i:i + BLOCK].astype(np.int64))[None].to(DEVICE)
            y = torch.from_numpy(val[i + 1:i + BLOCK + 1].astype(np.int64))[None].to(DEVICE)
            losses.append(model(x, y)[1].item())
    l = sum(losses) / len(losses)
    print(f"\n[1] held-out val: {l:.3f} nats/token · ppl {np.exp(l):.1f} · "
          f"{l / cpt:.3f} nats/char · {l / cpt / np.log(2):.3f} bits/char")
else:
    print("\n[1] tokens.bin not found — skipping ppl")

# --------------------------------------------- 2-4. held-out Q/A + metrics ---
gens = []
if os.path.exists(args.held_out):
    rows = [json.loads(l) for l in open(args.held_out, encoding="utf-8")]
    picks = np.random.choice(len(rows), size=min(args.n_qa, len(rows)), replace=False)
    print(f"\n[2] held-out Q/A ({args.held_out}, {len(rows)} available):\n")

    @torch.no_grad()
    def gen(prompt: str, max_new: int) -> str:
        ids = encode_bpe(prompt).astype(np.int64)
        idx = torch.from_numpy(ids)[None, :].to(DEVICE)
        for _ in range(max_new):
            logits = model(idx[:, -BLOCK:])[0][:, -1, :]
            probs = F.softmax(logits / args.temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return decode_bpe(idx[0].tolist())[len(prompt):]

    for k in picks:
        r = rows[int(k)]
        out = gen("Q: " + r["q"] + "\nA:", args.gen_tokens)
        gens.append(out)
        print(f"Q: {r['q'][:150]}{'…' if len(r['q']) > 150 else ''}")
        print(f"A: {out.strip()[:400]}\n")

    sp = sum(1 for g in gens if re.search(r"(mr\. speaker|monsieur le pr[ée]sident)",
                                          g[:160], re.I))
    words = " ".join(gens).lower().split()
    bigrams = list(zip(words, words[1:]))
    d2 = len(set(bigrams)) / max(1, len(bigrams))
    avg_len = sum(len(g.strip()) for g in gens) / max(1, len(gens))
    print(f"[3] register: {sp}/{len(gens)} start with \"Mr. Speaker\" · "
          f"avg length {avg_len:,.0f} chars")
    print(f"[4] repetition: distinct-2 = {d2:.2f} (1.0 = no repeated bigrams)")
else:
    print(f"\n[2] {args.held_out} not found — skipping Q/A eval")
