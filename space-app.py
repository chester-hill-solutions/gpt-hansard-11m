import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

CKPT = hf_hub_download("NathanielArfin/gpt-hansard-11m", "gpt-11m-sft-en-v3.pt")
ck = torch.load(CKPT, map_location="cpu")
cfg = ck["config"]
V, BLOCK = cfg["V"], cfg["block"]
TOP_K = 40

mm = {(a, b): ix for a, b, ix in ck["merges"]}
tok_bytes = {i: bytes([i]) for i in range(256)}
for a, b, ix in ck["merges"]:
    tok_bytes[ix] = tok_bytes[a] + tok_bytes[b]


def encode_bpe(text):
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
    return ids.astype(np.int64)


def decode_bpe(ids):
    return b"".join(tok_bytes[int(i)] for i in ids).decode("utf-8", errors="replace")


class Head(nn.Module):
    def __init__(self, n_embd, hs):
        super().__init__()
        self.key = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        att = att.masked_fill(
            torch.tril(torch.ones(x.shape[-2], x.shape[-2])) == 0, float("-inf"))
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
                                 nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = BLOCK
        self.tok_emb = nn.Embedding(V, cfg["n_embd"])
        self.pos_emb = nn.Embedding(BLOCK, cfg["n_embd"])
        self.blocks = nn.Sequential(*[Block(cfg["n_embd"], cfg["n_head"])
                                      for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], V)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        T = idx.shape[-1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
        return self.head(self.ln_f(self.blocks(x)))


model = GPT()
model.load_state_dict(ck["model"])
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())


@torch.no_grad()
def generate(prompt, temperature, max_new):
    if not prompt.strip():
        return "— give me a prompt (a Question Period question works well) —"
    idx = torch.from_numpy(encode_bpe(prompt))[None, :]
    for _ in range(int(max_new)):
        logits = model(idx[:, -BLOCK:])[:, -1, :]
        if TOP_K:
            v, _ = torch.topk(logits, TOP_K)
            logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat([idx, torch.multinomial(
            F.softmax(logits / max(0.05, temperature), dim=-1), 1)], dim=1)
    return decode_bpe(idx[0].tolist())


with gr.Blocks(title="Hansard Chat") as demo:
    gr.Markdown(
        f"""# Hansard Chat
**GPT-Hansard-11M** — a GPT built from scratch ({N_PARAMS:,} params) and trained
on 20 years of the Canadian House of Commons (Hansard, 2006–2026), then
fine-tuned on 70,808 real Question Period exchanges.
It speaks the register of Question Period in English (the EN version of the
model — French prompts stay in the letter of the law, not the spirit).
Everything it knows came from the parliamentary record — nothing else.
Dataset: `NathanielArfin/canadian-hansard-2006-now`""")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt",
                                value="Q: What is prorogation?\nA:",
                                lines=3)
            temp = gr.Slider(0.2, 1.5, value=0.6, label="Temperature")
            max_new = gr.Slider(30, 300, value=140, step=10, label="Max new tokens")
            btn = gr.Button("Speak, Minister")
        with gr.Column():
            out = gr.Textbox(label="The model", lines=14)
    btn.click(generate, [prompt, temp, max_new], out)
    gr.Examples([
        ["Q: What is prorogation?\nA:"],
        ["Q: What is a filibuster?\nA:"],
        ["Qu'est-ce que le Hansard ?\nR :"],
        ["Mr. Speaker, the cost of housing"],
        ["Bill C-"],
    ], [prompt])

demo.launch()
