# Morning run — capstone on the 2006–now corpus

## Built overnight
- `hansard-2006-now.txt` — 806,202,795 chars, 12 sessions (39-1 → 45-1 + reused 44-1; 42-1 spliced past its 400-sitting cap)
- `merges.json` — 1,024 merges, **vocab 1280** (trained on an era-sampled slice)
- `tokens.bin` — **258,174,079 tokens** (uint16, 516 MB) at 3.12 chars/token
- `tokens-info.json` — per-session table
- Old 44-1 tokenizer backed up as `merges-441.json` / `tokens-441.bin`

## Colab (T4 runtime)
1. Upload three files from this folder: `gpt_tokens.py`, `merges.json`, `tokens.bin`
   (516 MB — the only big one)
2. Run:
```
!python gpt_tokens.py --preset default --sft --steps 20000 --batch 16
```
   - step-0 loss should read ≈ 7.15 (= ln 1280)
   - 20,000 steps × 16 × 512 = ~164M tokens · ETA ~8–11 h on T4 (overnight)
   - shorter session? `--steps 12000` ≈ ~98M tokens, ~5–7 h
3. Plan B (if tokens.bin upload stalls): upload `hansard-2006-now.txt` instead and
   add `--max-encode-chars 999999999` (re-encodes ~25 min on Colab before training)

## Bring back
Loss curve + the three `[prompt]/[model]` samples + SFT Q/A answers.
We then compute **bits per character** vs the char model's 1.378 nats/char —
the fair cross-tokenization verdict.
