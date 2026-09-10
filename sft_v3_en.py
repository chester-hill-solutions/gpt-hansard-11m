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
GPT-0 Academy — SFT v3: EN-locked QP fine-tune, trained long enough to matter.

Diagnosis it fixes (probe_garble loop, Sep 8): the shipped SFT runs did 300
steps x 12 x 512 = 1.84M tokens over a 36.5M-token stream = 0.05 epochs, at a
flat LR with no schedule — the model learned the name-block pattern but not
the Q->A mapping, and the bilingual mix contaminated EN outputs (EN-only
tokenizer byte-falls-back French). Language-locking is per-checkpoint here.

Changes vs gpt_tokens.py's SFT stage:
  - EN pairs only (sft/qp-pairs.jsonl, lang=="en") -> own stream cache
  - fresh start from gpt-11m-base.pt (not the FR-drifted SFT checkpoint)
  - 3,000 steps (~1.6 epochs) with warmup + cosine decay
  - logs sft loss + pretrain-replay loss + held-out val loss (forgetting gauge)

Run:  uv run sft_v3_en.py 2>&1 | tee sft-v3-en.log
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config --
CKPT = "gpt-11m-base.pt"
OUT = "gpt-11m-sft-en-v3.pt"
SFT_DATA = "sft/qp-pairs.jsonl"
STREAM_CACHE = "sft-stream-en.bin"
BLOCK, BATCH = 512, 12
STEPS = 3000
WARMUP = 100
MAX_LR, MIN_LR = 1e-5, 1e-6
WEIGHT_DECAY = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ------------------------------------------------------------- tokenizer ----
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ck["config"]
raw_merges = ck["merges"]
merges = {(a, b): ix for a, b, ix in raw_merges}
tok_bytes = {i: bytes([i]) for i in range(256)}
for (a, b), ix in merges.items():
    tok_bytes[ix] = tok_bytes[a] + tok_bytes[b]
V = 256 + len(merges)

def encode_bpe(text: str) -> np.ndarray:
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in merges.items():
        hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if hit.size == 0:
            continue
        if a == b:
            keep, last = [], -2
            for i in hit:
                if i > last + 1:
                    keep.append(i); last = i
            keep = np.array(keep)
        else:
            keep = hit
        ids[keep] = idx
        ids = np.delete(ids, keep + 1)
    return ids.astype(np.uint16)

def decode_bpe(ids) -> str:
    return b"".join(tok_bytes[int(i)] for i in ids).decode("utf-8", errors="replace")

# ------------------------------------------------------------------- data ----
# EN-locked SFT stream, sentinel-packed exactly like gpt_tokens.py
if os.path.exists(STREAM_CACHE):
    stream_np = np.fromfile(STREAM_CACHE, dtype=np.uint16).astype(np.int64)
    labels_np = np.fromfile(STREAM_CACHE + ".labels", dtype=np.int32).astype(np.int64)
    print(f"SFT stream: {len(stream_np):,} tokens (cached)")
else:
    PAIRS = [(r["q"], f"{r['who_a']}: {r['a']}")
             for r in map(json.loads, open(SFT_DATA, encoding="utf-8"))
             if r.get("lang") == "en" and r.get("q") and r.get("a")]
    print(f"SFT data: {len(PAIRS):,} EN pairs from {SFT_DATA}")
    big = "".join(f"\x00Q: {q}\nA:\x00 {a}\n" for q, a in PAIRS)
    ids = encode_bpe(big).astype(np.int64)
    z = np.flatnonzero(ids == 0)
    bounds = np.concatenate((z[1:], [len(ids)]))
    segs = [s for s in (ids[b + 1:e] for b, e in zip(z, bounds)) if len(s) > 0]
    if len(segs) % 2 != 0:
        raise SystemExit(f"sentinel packing produced {len(segs)} segments (expected even)")
    stream_np = np.concatenate(segs)
    labels_np = np.concatenate([
        np.full(len(s), -100, dtype=np.int64) if i % 2 == 0 else s
        for i, s in enumerate(segs)])
    stream_np.astype(np.uint16).tofile(STREAM_CACHE)
    labels_np.astype(np.int32).tofile(STREAM_CACHE + ".labels")
    print(f"SFT stream: {len(stream_np):,} tokens (packed, {len(PAIRS):,} EN pairs, "
          f"prompts masked) — {STEPS * BATCH * BLOCK / max(1, len(stream_np)):.2f} epochs "
          f"over the stream at {STEPS} steps")

stream_t = torch.from_numpy(stream_np)
labels_t = torch.from_numpy(labels_np)
N = len(stream_t)

# replay + val streams from the EN corpus tokens (forgetting gauges)
all_ids = np.fromfile("tokens.bin", dtype=np.uint16)
split = int(len(all_ids) * 0.99)

def sft_batch():
    ix = torch.randint(max(1, N - BLOCK), (BATCH,))
    xb = stream_t[ix[:, None] + torch.arange(BLOCK)].to(DEVICE)
    yb = labels_t[ix[:, None] + 1 + torch.arange(BLOCK)].to(DEVICE)   # shifted targets
    return xb, yb

def lm_batch(src, batch=6):
    ix = np.random.randint(0, len(src) - BLOCK - 1, batch)
    x = torch.stack([torch.from_numpy(src[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(src[i + 1:i + BLOCK + 1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ------------------------------------------------------------------ model ---
class Head(nn.Module):
    def __init__(self, n_embd, hs):
        super().__init__()
        self.key = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
    def forward(self, x):
        att = self.query(x) @ self.key(x).transpose(-2, -1) * self.key(x).shape[-1] ** -0.5
        att = att.masked_fill(torch.tril(torch.ones(x.shape[1], x.shape[1])) == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ self.value(x)

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
        return x + self.mlp(self.ln2(x))

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
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.reshape(-1), ignore_index=-100)
        return logits, loss

model = GPT(cfg["V"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block"]).to(DEVICE)
model.load_state_dict(ck["model"])
print(f"model: {sum(p.numel() for p in model.parameters()):,} params · {DEVICE} · "
      f"from {CKPT} · {STEPS} steps")

# resume support: pick up from the periodic checkpoint if one exists
start_step = 0
if os.path.exists(OUT):
    rck = torch.load(OUT, map_location=DEVICE, weights_only=False)
    if rck.get("config") == cfg:
        model.load_state_dict(rck["model"])
        start_step = rck.get("sft_step", 0)
        print(f"resumed from {OUT} at step {start_step}")
    else:
        print(f"{OUT} exists but config mismatch — starting fresh")

def lr_at(step):
    if step < WARMUP:
        return MAX_LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * t))

@torch.no_grad()
def sample(prompt, max_new=90):
    model.eval()
    ids = torch.from_numpy(encode_bpe(prompt).astype(np.int64))[None].to(DEVICE)
    for _ in range(max_new):
        logits = model(ids[:, -BLOCK:])[0][:, -1, :]
        ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
    model.train()
    return decode_bpe(ids[0].tolist())[len(prompt):]

opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
model.train()
for step in range(start_step, STEPS):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    xb, yb = sft_batch()
    if (yb != -100).sum() == 0:
        continue
    _, loss = model(xb, yb)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0 or step == STEPS - 1:
        model.eval()
        s = model(*sft_batch())[1].item()
        r = model(*lm_batch(all_ids[:split]))[1].item()
        v = model(*lm_batch(all_ids[split:]))[1].item()
        model.train()
        print(f"sft step {step:4d} · task {s:.3f} · replay {r:.3f} · val {v:.3f} "
              f"· lr {lr_at(step):.2e}", flush=True)
        if step % 500 == 0 or step == STEPS - 1:
            print(f"  sample: {sample('Q: What is Hansard?\nA:')!r}", flush=True)
    if step % 500 == 0 or step == STEPS - 1:      # survive server restarts
        torch.save({"model": model.state_dict(), "merges": raw_merges, "config": cfg,
                    "sft": True, "sft_data": SFT_DATA, "sft_steps": STEPS,
                    "sft_step": step}, OUT)

torch.save({"model": model.state_dict(), "merges": raw_merges, "config": cfg,
            "sft": True, "sft_data": SFT_DATA, "sft_steps": STEPS}, OUT)
print(f"saved -> {OUT}")

for q in ["What is Hansard?", "What is prorogation?", "What is a filibuster?",
          "What is the excuse for this government's inaction on Faries?"]:
    print(f"\nQ: {q}\nA: {sample('Q: ' + q + '\nA:', max_new=140)}")
