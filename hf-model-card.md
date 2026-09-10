---
license: apache-2.0
language:
- en
- fr
datasets:
- NathanielArfin/canadian-hansard-2006-now
tags:
- gpt
- from-scratch
- hansard
- parliament
- education
---

# GPT-Hansard-11M

A GPT trained **from scratch** on the official Debates (Hansard) of the House
of Commons of Canada, 2006–2026. The tokenizer, the weights, and the training
stack all start from zero.

## Architecture

| | |
|---|---|
| Parameters | 11,330,048 (tied embeddings) |
| Layers / heads / d_model | 6 / 6 / 384 |
| Context | 512 tokens (~1,600 characters) |
| Vocab | 1,280 (256 bytes + 1,024 BPE merges) |
| Init | GPT-2-style (std 0.02), zero-init residual projections |

## Training

**Pretraining:** 20,000 steps · batch 16 × block 512 · AdamW · cosine LR
6e-4 → 6e-5 with 200-step warmup · grad clip 1.0 · ~164M tokens seen
(0.64 epochs of the 258M-token English stream).

**SFT:** prompt-masked, packed Q/A stream from **70,808 real Question Period
exchanges** mined from the official XML (`Type="Question"`/`Type="Answer"`
labels — no heuristics), language-locked: French questions pair only with
French answers, English with English. Answer targets open with the responder's
Hansard attribution, so generations lead with a name, role, and party.
300 steps at lr 1e-5.

## Measured results

Held-out tail of the corpus (never trained on):

| Model | Params | Bits/char | QP register |
|---|---|---|---|
| Char-level GPT (baseline) | 242K | 1.99 | — |
| Token GPT, small | 989K | 1.334 | in-register |
| **Token GPT, this model** | **11.33M** | **0.918** | **5/6 attribution + speaker form** |

SFT did not degrade pretrained knowledge: held-out perplexity after SFT sits
at the base model's level (2.03 nats/token), while Q/A format generalizes to
questions never in the training pairs.

## Scope

Closed-book: everything it knows came from 806M characters of Hansard, and its
content is exactly what 11M parameters can carry. It speaks Parliament's form
fluently (Standing Orders, tabling formulas, the ministerial non-answer), opens
with the responder's attribution, and usually answers in the language of the
question, though it occasionally crosses languages. It also invents bill
numbers, dates, statistics, and names with complete confidence; treat every
factual claim as unverified.

## Usage

```python
import torch, json
import numpy as np

ck = torch.load("gpt-11m-sft-bilingual-named.pt", map_location="cpu")
# model: 6-layer GPT, d=384, block 512 — the trainer is gpt_tokens.py
```

Trainer and full pipeline: `gpt_tokens.py` (self-contained, PEP 723).

## Provenance & rights
- **Corpus:** Official Debates (Hansard), House of Commons of Canada,
  parliaments 39-1 → 45-1, from the official XML. © Crown copyright.
- **SFT pairs:** mined from the same XML with ground-truth labels.
- **Weights:** Apache-2.0, by Nathaniel Arfin. Not affiliated with or
  endorsed by the House of Commons.

## Companion

- Dataset: `NathanielArfin/canadian-hansard-2006-now`
