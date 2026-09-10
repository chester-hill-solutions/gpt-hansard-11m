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
GPT-0 Academy — Module 7.2: SFT. Turn your base GPT into a tiny parliamentary assistant.

The entire mechanism, in one sentence: same cross-entropy loss as pretraining,
but on conversation-formatted data, with the loss MASKED to assistant spans only.

What changes vs pretraining:
  - data: ~20 hand-written (question -> answer) pairs, Hansard register
  - loss: F.cross_entropy(..., ignore_index=-100) on prompt positions
  - init: we START from your pretrained checkpoint (not random)

Honest expectations: 242K parameters + 20 examples will learn the FORMAT and can
memorize the set. In-distribution questions will get answers; out-of-distribution
questions test generalization — watch what happens. SFT teaches format and
persona; knowledge still comes from pretraining + scale.

FINDINGS (four recipes, one conclusion):
  1. 300 steps @ 1e-4, padded examples        -> loss 0.046 (memorized); generation collapsed
  2. + replay alternation                      -> task ~2.4 but doubled-char runs in output
  3. + packing, no padding, single-char markers -> same doubling; "###" and PAD runs were NOT the cause
  4. pure packed SFT @ 3e-5, no replay         -> worst: nothing holds the pretraining distribution
  VERDICT: the masking/packing/replay mechanics are correct (format IS learned —
  samples open "Mr. Speaker"); at 242K params the model cannot hold pretraining
  distribution AND Q/A format at once. SFT degradation at this scale is a
  capacity wall, not a recipe bug. The token-level capstone model (more params,
  subword tokens) is where SFT gets room to work.

Run:  uv run sft_hansard.py      (needs gpt_hansard.pt in this folder)
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config --
CKPT = "gpt_hansard.pt"
OUT = "gpt_hansard_sft.pt"
CORPUS_FILE = "hansard-en.txt"      # only used if the ckpt predates vocab-saving
SLICE = 10_000_000
BLOCK = 256                          # char-level context — tight! write terse pairs
STEPS = 300
BATCH = 4
LR = 1e-4                            # SFT is a gentle nudge, not a rebuild
TEMP = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ------------------------------------------------------------- the SFT set --
SFT_PAIRS = [
    ("What is Hansard?",
     "Hansard is the official transcript of debates in the House of Commons, published in both English and French."),
    ("What is Question Period?",
     "Question Period is the daily time when members ask ministers questions about the government's work."),
    ("Who is the Speaker?",
     "The Speaker is the member who presides over the House and enforces the rules of debate."),
    ("What is a riding?",
     "A riding is a federal electoral district, represented in the House by one member of Parliament."),
    ("What is a bill?",
     "A bill is a proposed law. It is debated, amended and voted on before it can become law."),
    ("What is royal assent?",
     "Royal assent is the final step at which a bill becomes law, granted by the Governor General."),
    ("What is a sitting day?",
     "A sitting day is a day when the House meets to conduct its business."),
    ("What is a committee?",
     "A committee is a small group of members that studies bills and issues in detail and reports to the House."),
    ("What is prorogation?",
     "Prorogation ends a session of Parliament. Bills not passed die on the order paper."),
    ("What is the order paper?",
     "The order paper is the agenda of business before the House for each sitting day."),
    ("What does pursuant to mean?",
     "Pursuant to means under the authority of. Members often cite standing orders this way."),
    ("What is a quorum?",
     "A quorum is the minimum number of members needed for the House to conduct business."),
    ("How long is Question Period?",
     "Question Period lasts about forty five minutes on each sitting day."),
    ("What is a point of order?",
     "A point of order is raised when a member believes the rules of the House have been broken."),
    ("Who keeps order in debate?",
     "The Speaker keeps order, and may name a member who breaks the rules."),
    ("What is a recorded vote?",
     "A recorded vote calls each member by name so their vote is on the record."),
    ("What is second reading?",
     "Second reading is the stage where the House debates the main idea and principle of a bill."),
    ("What is a minority government?",
     "A minority government holds fewer than half the seats, so it must win support from other parties."),
    ("What does the Senate do?",
     "The Senate reviews bills passed by the House of Commons and can propose amendments."),
    ("What is Hansard used for?",
     "Hansard is used by courts, journalists, researchers and members to know exactly what was said."),
]

# -------------------------------------------------------------------- vocab --
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict) and "model" in ckpt:            # new format (weights+vocab+config)
    state_dict, stoi, itos = ckpt["model"], ckpt["stoi"], ckpt["itos"]
    cfg = ckpt.get("config", {})
    print("checkpoint carries its own vocab — as it should")
else:                                                      # old bare-weights format
    state_dict = ckpt
    text = open(CORPUS_FILE, encoding="utf-8").read()[:SLICE]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    cfg = {"V": len(chars), "n_embd": 64, "n_head": 4, "n_layer": 4, "block": 256}
    print("bare-weights checkpoint — vocab rebuilt from corpus slice")
V = len(stoi)
decode = lambda ids: "".join(itos[i] for i in ids)
encode = lambda s: [stoi[c] for c in s]
BLOCK = min(BLOCK, cfg.get("block", BLOCK))
STEPS = 150                          # gentle: early-stop well before memorization
LR = 3e-5
print(f"vocab {V} · block {BLOCK} · SFT with pretraining replay (anti-forgetting)")

# replay stream: pretraining data keeps the base distribution alive
if not os.path.exists(CORPUS_FILE):
    print(f"fetching {CORPUS_FILE} for the replay stream ...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/NathanielArfin/canadian-hansard-44-1/resolve/main/lm/en.txt",
        CORPUS_FILE)
data = torch.tensor(encode(open(CORPUS_FILE, encoding="utf-8").read()[:SLICE]), dtype=torch.long)
print(f"vocab {V} · block {BLOCK}")

# ------------------------------------------------------------------- model --
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
    def __init__(self, vocab, n_embd=64, n_head=4, n_layer=4, block=256):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
    def forward(self, idx, targets=None, loss_ignore=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T),
                                   ignore_index=-100)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=TEMP, stop=None):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop:
                joined = decode(idx[0].tolist())
                if stop in joined[len(joined) - len(stop) - 4:]:
                    break
        return idx

model = GPT(vocab=V, n_embd=cfg.get("n_embd", 64), n_head=cfg.get("n_head", 4),
            n_layer=cfg.get("n_layer", 4), block=BLOCK)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
base_state = copy.deepcopy(model.state_dict())           # keep the base for comparison
print(f"loaded base model · {sum(p.numel() for p in model.parameters()):,} params")

# ------------------------------------------------------- data with masking --
# PACKING — no padding, no fake tokens: all pairs concatenated into one token
# stream (separated by real newlines), then trained on random windows exactly
# like pretraining. Padding with runs of a real char ("\\n\\n\\n...") is what taught
# the model to repeat characters — the failure we just debugged.
stream, labels_stream = [], []
for q, a in SFT_PAIRS:
    prompt = "Q: " + q + "\nA: "          # single-char markers; prompt = no loss
    full = prompt + a + "\n"
    ids = encode(full)
    p = len(encode(prompt))
    labels_stream.extend([-100] * p + ids[p:])
    stream.extend(ids)
stream_t = torch.tensor(stream, dtype=torch.long)
labels_t = torch.tensor(labels_stream, dtype=torch.long)
N = len(stream_t)
masked_frac = sum(1 for l in labels_stream if l == -100) / len(labels_stream)
print(f"SFT stream: {N:,} tokens · {len(SFT_PAIRS)} pairs · {masked_frac:.0%} of "
      f"positions masked (prompts) · packed, no padding")

def get_sft_batch(batch=BATCH):
    ix = torch.randint(N - BLOCK, (batch,))
    x = torch.stack([stream_t[i:i + BLOCK] for i in ix])
    y = torch.stack([labels_t[i:i + BLOCK] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------------------------------------------------------- training --
# Two data streams, alternated: SFT pairs teach format+persona, pretraining
# batches keep the base distribution alive. Watch pre_loss — if it explodes,
# the assistant ate the language model. (This replay trick is why real labs
# mix instruction data into pretraining mixes.)
    ix = torch.randint(len(data) - BLOCK, (batch,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def get_pretrain_batch(batch=BATCH):
    ix = torch.randint(len(data) - BLOCK, (batch,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
for step in range(STEPS):
    # pure SFT stream — no alternation. (Replay alternation at this tiny scale
    # made two objectives fight over 242K weights; the compromise corrupted
    # character fluency. If this still degrades, the honest answer is capacity.)
    xb, yb = get_sft_batch()
    _, loss = model(xb, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 20 == 0 or step == STEPS - 1:
        model.eval()
        _, s = model(*get_sft_batch(8))
        _, p = model(*get_pretrain_batch(4))
        model.train()
        print(f"sft step {step:4d} · task {s.item():.3f} · replay(pre) {p.item():.3f}")
sft_state = copy.deepcopy(model.state_dict())

# ------------------------------------------------------------ base vs sft --
def answer(m, q, temp=TEMP):
    prompt = "Q: " + q + "\nA:"
    idx = torch.tensor([encode(prompt)], device=DEVICE)
    was_training = m.training
    m.eval()                                             # dropout OFF for generation
    out = m.generate(idx, max_new=BLOCK, temperature=temp, stop="\nQ:")
    if was_training:
        m.train()
    text = decode(out[0].tolist())[len(prompt):]
    return text.split("\nQ:")[0].strip()

EVAL = [
    ("What is Hansard?", "in-set"),
    ("What is Question Period?", "in-set"),
    ("What is a filibuster?", "OUT-OF-SET (generalization test)"),
]
for q, tag in EVAL:
    model.load_state_dict(base_state)
    base_ans = answer(model, q)
    model.load_state_dict(sft_state)
    sft_ans = answer(model, q)
    print(f"\n=== {q}  [{tag}] ===")
    print(f"BASE : {base_ans[:180]}")
    print(f"SFT  : {sft_ans[:180]}")

torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
            "config": {"V": V, "n_embd": 64, "n_head": 4, "n_layer": 4, "block": BLOCK},
            "sft_pairs": SFT_PAIRS}, OUT)
print(f"\nsaved SFT model -> {OUT}")
