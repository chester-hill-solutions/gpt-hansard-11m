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
GPT-0 Academy — prompt your base model interactively.

This is a BASE model (a completer, not an assistant): it continues whatever
you type in the register of Canadian Hansard. Prompting = conditioning; the
quality of the continuation shows you exactly what "val loss 1.3" knows.

Usage:
  uv run prompt_hansard.py                     # interactive REPL
  uv run prompt_hansard.py --prompt "Mr. Speaker, the cost of housing" --temperature 0.8
  uv run prompt_hansard.py --prompt "Bill C-" --temperature 0.5 --max-new 200

REPL commands: temp=0.9 · new=300 · exit
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = "gpt_hansard.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------- load --
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict) and "model" in ckpt:            # new format: weights+vocab
    state_dict, stoi, itos, cfg = ckpt["model"], ckpt["stoi"], ckpt["itos"], ckpt["config"]
    print("checkpoint carries its own vocab")
else:                                                      # old bare-weights format
    state_dict = ckpt
    text = open("hansard-en.txt", encoding="utf-8").read()[:10_000_000]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    cfg = {"V": len(chars), "n_embd": 64, "n_head": 4, "n_layer": 4, "block": 256}
    print("bare-weights checkpoint — vocab rebuilt from corpus slice")
decode = lambda ids: "".join(itos[i] for i in ids)
encode = lambda s: [stoi[c] for c in s]
BLOCK = cfg.get("block", 256)
print(f"loaded {CKPT} · vocab {len(stoi)} · block {BLOCK} · device {DEVICE}")

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
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHead(n_embd, n_head)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(0.1))
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
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        return self.head(x)

model = GPT(cfg["V"], cfg.get("n_embd", 64), cfg.get("n_head", 4),
            cfg.get("n_layer", 4), BLOCK).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()  # dropout OFF — the silent sampler-killer, remembered forever

def sanitize(prompt: str) -> str:
    """the vocab comes from a 10M-char slice; drop chars it has never seen"""
    keep = [c for c in prompt if c in stoi]
    dropped = len(prompt) - len(keep)
    if dropped:
        print(f"  (dropped {dropped} chars outside the model's 206-char vocab)")
    return "".join(keep)

def generate(prompt: str, temperature=0.8, max_new=250, top_k=0):
    ids = torch.tensor([encode(sanitize(prompt))], device=DEVICE)
    for _ in range(max_new):
        logits = model(ids[:, -BLOCK:])[:, -1, :]
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits / temperature, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
    return decode(ids[0].tolist())[len(sanitize(prompt)):]

# ------------------------------------------------------------------- REPL --
ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default=None)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--max-new", type=int, default=250)
ap.add_argument("--top-k", type=int, default=0)
ap.add_argument("--seed", type=int, default=None)
args = ap.parse_args()
if args.seed is not None:
    torch.manual_seed(args.seed)

if args.prompt is not None:
    print("=== continuation ===")
    print(args.prompt + generate(args.prompt, args.temperature, args.max_new, args.top_k))
else:
    temp, new, k = args.temperature, args.max_new, args.top_k
    print("type a prompt; empty line quits; temp=0.9 new=300 k=20 to adjust")
    while True:
        try:
            p = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not p.strip():
            break
        if p.startswith("temp="): temp = float(p[5:]); print(f"temperature {temp}"); continue
        if p.startswith("new="): new = int(p[4:]); print(f"max_new {new}"); continue
        if p.startswith("k="): k = int(p[2:]); print(f"top_k {k}"); continue
        print(generate(p, temp, new, k))
print("done — class dismissed 🏛️")
