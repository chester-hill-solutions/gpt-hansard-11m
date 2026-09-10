# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5"]
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
GPT-0 Academy — your GPT, trained on Canadian Hansard.
Complete, self-contained training script (Module 5, block=256 config).

Run with uv (auto-installs the CPU torch wheel into an ephemeral env):
    uv run gpt_hansard.py
Or in Colab (T4 recommended): just run it — cuda is detected automatically.

Run in Colab (T4 recommended) or locally:
    1. Fetch the corpus (public, no token needed):
       !curl -sL -o hansard-en.txt https://huggingface.co/datasets/NathanielArfin/canadian-hansard-44-1/resolve/main/lm/en.txt
    2. python gpt_hansard.py

Trajectory expectation (Hansard, 10M-char slice, this config):
    step 0 ≈ 5.4 (≈ ln(vocab)) → 500 ≈ 2.3-2.4 → 1000 ≈ 1.9 → 5000 ≈ 1.6-1.8

Fixed vs the lesson listing: the causal mask is now derived from the actual
sequence length T instead of a hardcoded 128x128 buffer — the model now
accepts ANY block size. (Lesson: never hardcode what a tensor can tell you.)
"""

import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- config --
CORPUS_URL = (
    "https://huggingface.co/datasets/NathanielArfin/canadian-hansard-44-1/"
    "resolve/main/lm/en.txt"
)
CORPUS_FILE = "hansard-en.txt"
SLICE = 10_000_000      # chars to train on (RAM: long tensor = 8 B/char)
BLOCK = 256             # context length (the lever we doubled)
BATCH = 32
STEPS = 5000
LR = 1e-3
EVAL_EVERY = 500
N_EMBD, N_HEAD, N_LAYER = 64, 4, 4
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ------------------------------------------------------------------ data --
import os
if not os.path.exists(CORPUS_FILE):
    print(f"fetching {CORPUS_FILE} ...")
    urllib.request.urlretrieve(CORPUS_URL, CORPUS_FILE)

text = open(CORPUS_FILE, encoding="utf-8").read()
text = text[:SLICE]
print(f"corpus: {len(text):,} chars")

chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}   # note the unpack order — items() yields (char, index)!
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"vocab: {V} chars · expected initial loss ≈ ln({V}) = {math.log(V):.2f}")

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split, batch=BATCH, block=BLOCK):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block, (batch,))
    x = torch.stack([d[i:i + block] for i in ix])
    y = torch.stack([d[i + 1:i + block + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ----------------------------------------------------------------- model --
class Head(nn.Module):
    """one self-attention head — causal mask derived from T, never hardcoded"""
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        tril = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(tril == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return att @ v

class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    """communicate (attention) then compute (MLP), each with a residual path"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHead(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(0.1),
        )

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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=500, temperature=1.0):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# -------------------------------------------------------------- training --
model = GPT(vocab=V).to(DEVICE)
print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters on {DEVICE}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# --- optional, for the NEXT run (Lesson 5.4 lever #1): cosine LR schedule --
# max_lr, min_lr = 1e-3, 1e-4
# def lr_at(step):
#     if step < 100: return max_lr * (step + 1) / 100
#     t = (step - 100) / (STEPS - 100)
#     return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))
# and inside the loop, before opt.step():
#     for g in opt.param_groups: g["lr"] = lr_at(step)
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

for step in range(STEPS):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        xb, yb = get_batch("val")
        _, val_loss = model(xb, yb)
        model.train()
        print(f"step {step:5d} · train {loss.item():.3f} · val {val_loss.item():.3f}")

torch.save(
    {"model": model.state_dict(), "stoi": stoi, "itos": itos,
     "config": {"V": V, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "block": BLOCK}},
    "gpt_hansard.pt",
)
print("saved checkpoint -> gpt_hansard.pt  (weights + vocab + config — a checkpoint IS its tokenizer)")

# ------------------------------------------------------------ generation --
model.eval()
idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)   # newline = "beginning"
sample = decode(model.generate(idx, max_new=400, temperature=0.8)[0].tolist())
print("\n=== sample @ temperature 0.8 ===")
print(sample)
