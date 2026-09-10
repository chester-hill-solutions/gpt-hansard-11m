import torch, torch.nn as nn, torch.nn.functional as F
ck = torch.load("gpt_hansard_sft.pt", map_location="cpu", weights_only=False)
stoi, itos, cfg = ck["stoi"], ck["itos"], ck["config"]
decode = lambda ids: "".join(itos[i] for i in ids)
encode = lambda s: [stoi[c] for c in s]
BLOCK = cfg["block"]

class Head(nn.Module):
    def __init__(s, e, h):
        super().__init__(); s.key = nn.Linear(e, h, bias=False); s.query = nn.Linear(e, h, bias=False); s.value = nn.Linear(e, h, bias=False)
    def forward(s, x):
        B, T, C = x.shape
        k, q, v = s.key(x), s.query(x), s.value(x)
        a = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        a = a.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        return F.softmax(a, dim=-1) @ v
class MH(nn.Module):
    def __init__(s, e, h):
        super().__init__(); hs = e // h; s.heads = nn.ModuleList([Head(e, hs) for _ in range(h)]); s.proj = nn.Linear(e, e)
    def forward(s, x): return s.proj(torch.cat([hd(x) for hd in s.heads], dim=-1))
class Blk(nn.Module):
    def __init__(s, e, h):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(e), nn.LayerNorm(e); s.attn = MH(e, h)
        s.mlp = nn.Sequential(nn.Linear(e, 4*e), nn.GELU(), nn.Linear(4*e, e), nn.Dropout(0.1))
    def forward(s, x):
        x = x + s.attn(s.ln1(x)); return x + s.mlp(s.ln2(x))
class GPT(nn.Module):
    def __init__(s, v, e=64, h=4, l=4, b=256):
        super().__init__(); s.block = b; s.tok_emb = nn.Embedding(v, e); s.pos_emb = nn.Embedding(b, e)
        s.blocks = nn.Sequential(*[Blk(e, h) for _ in range(l)]); s.ln_f = nn.LayerNorm(e); s.head = nn.Linear(e, v)
    def forward(s, idx):
        B, T = idx.shape
        x = s.tok_emb(idx) + s.pos_emb(torch.arange(T))
        x = s.ln_f(s.blocks(x)); return s.head(x)

m = GPT(cfg["V"]); m.load_state_dict(ck["model"]); m.eval()

def gen_argmax(q, n=90):
    idx = torch.tensor([encode("Q: " + q + "\nA:")])
    for _ in range(n):
        logits = m(idx[:, -BLOCK:])[:, -1, :]
        idx = torch.cat([idx, logits.argmax(1, keepdim=True)], dim=1)
    return decode(idx[0].tolist())[len("Q: " + q + "\nA:"):]

def gen_top(q, T=0.7, n=90):
    idx = torch.tensor([encode("Q: " + q + "\nA:")])
    for _ in range(n):
        logits = m(idx[:, -BLOCK:])[:, -1, :]
        probs = F.softmax(logits / T, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return decode(idx[0].tolist())[len("Q: " + q + "\nA:"):]

q = "What is Hansard?"
print("ARGMAX :", repr(gen_argmax(q))[:200])
print("T=0.7  :", repr(gen_top(q, 0.7))[:200])
# also: top-5 at the first response position
idx = torch.tensor([encode("Q: " + q + "\nA:")])
logits = m(idx)[:, -1, :]
top = torch.topk(logits, 5)
print("top5 first-token:", [(itos[i], round(float(p), 3)) for i, p in zip(top.indices[0], top.values[0])])
