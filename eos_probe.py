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
"""EOS probe — does the v3 model emit its own trained separator (\x00) when
allowed, or run to budget? Decides the sentence-conclusion fix."""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))
        x = self.ln_f(self.blocks(x))
        return self.head(x)


ck = torch.load("gpt-11m-sft-en-v3.pt", map_location="cpu")
cfg = ck["config"]
mm = {(a, b): i for a, b, i in ck["merges"]}
tb = {i: bytes([i]) for i in range(256)}
for a, b, i in ck["merges"]:
    tb[i] = tb[a] + tb[b]


def enc(t):
    ids = np.frombuffer(t.encode(), dtype=np.uint8).astype(np.int32)
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
    return ids.astype(np.int64)


def dec(ids):
    return b"".join(tb[int(i)] for i in ids).decode("utf-8", errors="replace")


model = GPT(cfg["V"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block"])
model.load_state_dict(ck["model"])
model.eval()

Q = "Q: What is the excuse for this government's inaction on Faries?\nA:"
base_ids = enc(Q).tolist()
stats = {"sep_ended": 0, "qm_ended": 0, "budget": 0, "tails": []}
BUDGET = 160
for seed in range(12):
    torch.manual_seed(seed)
    ids = torch.from_numpy(np.array(base_ids, dtype=np.int64))[None, :]
    ended = None
    for _ in range(BUDGET):
        logits = model(ids[:, -cfg["block"]:])[:, -1, :].clone()
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = int(torch.multinomial(F.softmax(logits / 0.6, dim=-1), 1))
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == 0:
            ended = "SEP"
            break
    text = dec(ids[0].tolist())[len(Q):].replace("\x00", "<SEP>")
    if ended == "SEP":
        stats["sep_ended"] += 1
    elif "\nQ:" in text:
        stats["qm_ended"] += 1
    else:
        stats["budget"] += 1
    stats["tails"].append(
        f"seed{seed} {ended or 'BUDGET'}: ...{text[-90:].replace(chr(10), '\\\\n')}")
print(json.dumps(stats, indent=1))
print("\n".join(stats["tails"]))