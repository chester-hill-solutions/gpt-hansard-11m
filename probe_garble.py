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
[DEBUG-gb1] Garble probe — deterministic feedback loop for "the way the answers
are coming out" on the 11M Hansard checkpoints.

Checks, in one run:
  A. CODEBOOK: decode the head of sft-stream.bin with the checkpoint's own
     merges AND with merges-bilingual.json — count "Q: " markers and mojibake.
     A mismatch means the SFT stream was encoded under a different BPE
     codebook than the checkpoint uses at generation time.
  B. GENERATION: greedy (deterministic) + seeded T=0.7 samples for an EN
     probe, an FR probe, and the user's exact garble prompt.
  C. METRICS per output: French density, distinct-2, register-start rate.

Usage:  uv run /tmp/opencode/probe_garble.py            (from the repo dir)
"""
import json
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SFT_STREAM = "sft-stream.bin"
ALT_MERGES = "merges-bilingual.json"
PROMPTS = {
    "EN  ": "Q: What is Hansard?\nA:",
    "FR  ": "Q: Qu'est-ce que la prorogation ?\nA:",
    "USER": "Q: What is the excuse for this government's inaction on Faries?\nA:",
}
CKPTS = ["gpt-11m-base.pt", "gpt-11m-sft-en.pt",
         "gpt-11m-sft-bilingual.pt", "gpt-11m-sft-bilingual-named.pt",
         "gpt-11m-sft-en-v3.pt"]

# ------------------------------------------------------------- tokenizer ----
def load_merges(raw):
    mm = {(a, b): ix for a, b, ix in raw}
    tok_bytes = {i: bytes([i]) for i in range(256)}
    for (a, b), ix in mm.items():
        tok_bytes[ix] = tok_bytes[a] + tok_bytes[b]
    return mm, tok_bytes

def encode_bpe(text, mm):
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in mm.items():
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

def decode_bpe(ids, tok_bytes):
    return b"".join(tok_bytes[int(i)] for i in ids).decode("utf-8", errors="replace")

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
    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))
        x = self.ln_f(self.blocks(x))
        return self.head(x)

# ------------------------------------------------------------------ main ----
def fr_density(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    fr = [c for c in letters if c in "àâçéèêëîïôùûüœÀÂÇÉÈÊËÎÏÔÙÛÜŒ"]
    return len(fr) / len(letters)

def distinct2(text):
    words = text.lower().split()
    bg = list(zip(words, words[1:]))
    return len(set(bg)) / max(1, len(bg))

def register_start(text):
    return bool(re.search(r"(mr\. speaker|madam speaker|monsieur le pr[ée]sident"
                          r"|mme la pr[ée]sidente|l'hon\.|hon\.)", text[:120], re.I))

@torch.no_grad()
def gen(model, prompt, mm, tok_bytes, block, max_new=140, temperature=0.0, seed=1337):
    ids = encode_bpe(prompt, mm)
    idx = torch.from_numpy(ids.astype(np.int64))[None, :]
    if temperature > 0:
        torch.manual_seed(seed)
    for _ in range(max_new):
        logits = model(idx[:, -block:])[:, -1, :]
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return decode_bpe(idx[0].tolist(), tok_bytes)[len(prompt):]

print("=" * 78)
print("[A] CODEBOOK CHECK — decode head of sft-stream.bin under both merges")
stream = np.fromfile(SFT_STREAM, dtype=np.uint16)[:4000]
ck = torch.load(CKPTS[-1], map_location="cpu", weights_only=False)
own_raw = ck["merges"]
alt_raw = json.load(open(ALT_MERGES))
print(f"    checkpoint merges count: {len(own_raw)} · {ALT_MERGES}: {len(alt_raw)}")
same = (own_raw == alt_raw)
print(f"    checkpoint merges == bilingual merges: {same}")
own_mm, own_bytes = load_merges(own_raw)
alt_mm, alt_bytes = load_merges(alt_raw)
d_own, d_alt = decode_bpe(stream, own_bytes), decode_bpe(stream, alt_bytes)
q_own, q_alt = d_own.count("Q: "), d_alt.count("Q: ")
print(f'    decoded with ckpt merges : {q_own:2d} "Q: " markers · head: {d_own[:150]!r}')
print(f'    decoded with bilingual   : {q_alt:2d} "Q: " markers · head: {d_alt[:150]!r}')

print("=" * 78)
print("[B] GENERATION — greedy + seeded T=0.7 · 140 tokens")
red = []
for path in CKPTS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mm, tok_bytes = load_merges(ck["merges"])
    model = GPT(cfg["V"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block"])
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"\n--- {path} · V={cfg['V']} · block={cfg['block']} ---")
    for tag, prompt in PROMPTS.items():
        g = gen(model, prompt, mm, tok_bytes, cfg["block"], temperature=0.0)
        s = gen(model, prompt, mm, tok_bytes, cfg["block"], temperature=0.7)
        body = g.strip()
        m = dict(frd=fr_density(body), d2=distinct2(body), reg=register_start(body))
        print(f"[{tag}] GREEDY : {body[:170]!r}")
        print(f"[{tag}] T=0.7  : {s.strip()[:170]!r}")
        print(f"[{tag}] metrics greedy: fr={m['frd']:.2f} d2={m['d2']:.2f} reg={int(m['reg'])}")
        if tag == "EN  " and m["frd"] > 0.10:
            red.append(f"{path}: EN prompt -> FR density {m['frd']:.2f}")
        if m["d2"] < 0.20 and len(body.split()) >= 40:
            red.append(f"{path}: {tag} greedy distinct-2 {m['d2']:.2f} (repetition loop)")

print("=" * 78)
if same and q_own < 3:
    red.append("stream decodes cleanly under ckpt merges yet shows <3 Q-markers — inspect")
if not same and q_own < q_alt:
    red.append(f"CODEBOOK MISMATCH: stream reads better under bilingual merges "
               f"({q_alt} vs {q_own} Q-markers) but ckpt carries EN merges")
print("VERDICT:", "RED — " + "; ".join(red) if red else "GREEN")
sys.exit(1 if red else 0)
