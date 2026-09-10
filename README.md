# GPT-Hansard-11M

A GPT trained **from scratch** on the official Debates (Hansard) of the House
of Commons of Canada, 2006–2026, plus a bilingual variant. The tokenizer, the
weights, and the training stack all start from zero.

This is a *measured, never borrowed* lab: every claim about the models is
backed by the deterministic probe (`probe_garble.py`) or the eval suite
(`eval_hansard.py`). The working log of the project's history — including a
retracted-bug record and eval-suite-as-referee rules — lives in `PLAN.md`.

**Blobs don't live here.** The checkpoints, SFT streams, and mined pairs are
hosted on Hugging Face; this repo carries code, docs, and a manifest of
pointers (`ASSETS.md`). One command restores the full lab:

```bash
uv run fetch_assets.py
```

---

## Architecture

| | |
|---|---|
| Parameters | 11,330,048 (tied embeddings) |
| Layers / heads / d_model | 6 / 6 / 384 |
| Context | 512 tokens (~1,600 characters) |
| Vocab | 1,280 (256 bytes + 1,024 BPE merges) |
| Init | GPT-2-style (std 0.02), zero-init residual projections |

## Checkpoints (on Hugging Face)

| File | What it is |
|---|---|
| `gpt-11m-base.pt` | Pretrain, EN Hansard 2006–2026 |
| `gpt-11m-sft-en.pt` | SFT on ~35k mined EN Q→A pairs (best single-language model) |
| `gpt-11m-sft-bilingual.pt` | SFT on EN+FR pairs mixed |
| `gpt-11m-sft-bilingual-named.pt` | SFT with speaker-attribution prefixes (costs ~20 boilerplate tokens at this scale) |
| `gpt-11m-sft-en-v3.pt` | SFT v3: EN-only, 3,000 steps ≈ 1.5 epochs, warmup + cosine schedule — the fix for the garble diagnosed in the log |
| `sft-stream-en.bin` / `.labels` | Cached EN SFT stream (12,229,868 tokens, 35,401 pairs) — the SFT v3 corpus |
| `sft/qp-pairs.jsonl` | 70,808 mined question→answer pairs (`q, a, lang, src, topic, who_a, who_q`) |
| `sft/qp-heldout.jsonl` | Held-out Q/A pairs, never trained on (committed here — all the eval suite needs) |

- Model repo: `NathanielArfin/gpt-hansard-11m` → https://huggingface.co/NathanielArfin/gpt-hansard-11m
- SFT corpus: `NathanielArfin/gpt-hansard-11m-sft` → https://huggingface.co/datasets/NathanielArfin/gpt-hansard-11m-sft

## Quick start

```bash
uv run fetch_assets.py                         # restore checkpoints + SFT corpus from HF
uv run eval_hansard.py --ckpt gpt-11m-sft-en-v3.pt --n-qa 6
uv run probe_garble.py                         # deterministic red/green garble probe
uv run sft_v3_en.py                            # re-run SFT v3 (resumes from its periodic checkpoint)
```

All scripts are PEP 723: dependencies (CPU-only torch via the
`download.pytorch.org/whl/cpu` index) are declared in the script header, so
`uv run` builds the environment for you. No install step. `eval_hansard.py`
and `probe_garble.py` only need `gpt-11m-sft-en-v3.pt` plus `qp-heldout.jsonl`
(already in git).

## Reproducing the pretrain

The tokenized pretrain corpus (`tokens.bin`, ~493 MB) lives on the HF dataset
`NathanielArfin/canadian-hansard-2006-now`. Regenerate or re-fetch it and run

```bash
uv run gpt_tokens.py  # or upload to a Colab T4 per the HF model card
```

## Publishing to HF

Anytime the lab produces a new checkpoint or SFT dataset, point it back:

```bash
HF_TOKEN=... uv run publish_sft.py --from /path/to/your/local/lab
```

`publish_model.py` / `upload_hf.py` cover the base corpus and the Gradio
space. Tokens always come from `HF_TOKEN` (env), never written to disk.

## Serving

`model-service/` is a small FastAPI app exposing the model with attention-map
visualizations:

```bash
uv run fetch_assets.py                          # gets gpt-11m-sft-bilingual-named.pt too
docker build -f model-service/Dockerfile -t gpt-hansard-11m .
docker run -p 8787:8787 gpt-hansard-11m
```

## License & provenance

- **Code**: Apache-2.0 (see `LICENSE`).
- **Corpus**: Crown copyright (federal/provincial) or Open Government
  Licence - Canada. The full per-source compliance record, including the
  attribution conditions applied to every published artifact, is in
  `LEGAL.md` — no artifact ships without its row.

## Files

| File | Purpose |
|---|---|
| `gpt_tokens.py` | Pretrainer: BPE → tokens → GPT (canonical trainer) |
| `bpe_hansard.py` `train_merges_bilingual.py` | Byte-pair-encoding merges (EN / bilingual) |
| `sft_hansard.py` | Tiny-char SFT (⚠ carries a known unshifted-targets bug — see PLAN.md) |
| `sft_v3_en.py` | SFT v3 trainer: EN-locked, scheduled LR, replay/val gauges, periodic checkpoint + resume |
| `eval_hansard.py` | Eval suite: held-out ppl, held-out Q/A, register, distinct-2 |
| `probe_garble.py` | Deterministic red/green probe for answer quality (codebook check + generation metrics) |
| `eos_probe.py` | Measures whether the model ends answers itself or runs to budget (foregrounds the sentence-conclusion decode fix) |
| `mine_qp.py` | Mined the 70k Q→A pairs from Hansard |
| `prompt_hansard.py` `gen_local.py` `space-app.py` | Generation/demo paths |
| `publish_model.py` `publish_sft.py` `upload_hf.py` | Hugging Face publication (token via `HF_TOKEN`, never on disk) |
| `ASSETS.md` `fetch_assets.py` | Pointer manifest + one-command restore of all HF-hosted files |
| `PLAN.md` | The project roadmap and measured-never-borrowed record (verbose) |