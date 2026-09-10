import torch, torch.nn as nn, torch.nn.functional as F

# rebuild the EXACT vocab of the finished run (deterministic from the same slice)
text = open("hansard-en.txt", encoding="utf-8").read()[:10_000_000]
chars = sorted(set(text)); V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
decode = lambda ids: "".join(itos[i] for i in ids)

# model definition (same as the run)
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
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(0.1))
    def forward(self, x):
        x = x + self.attn(self.ln1(x)); x = x + self.mlp(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self, vocab, n_embd=64, n_head=4, n_layer=4, block=256):
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
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        return logits, None

ckpt = torch.load("gpt_hansard.pt", map_location="cpu", weights_only=True)
model = GPT(vocab=V)
model.load_state_dict(ckpt)
model.eval()
torch.manual_seed(97)
idx = torch.zeros((1, 1), dtype=torch.long)
for _ in range(400):
    logits, _ = model(idx[:, -256:])
    probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
    idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
print(decode(idx[0].tolist()))
