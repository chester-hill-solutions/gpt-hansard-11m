# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
GPT-0 Academy — overnight corpus expansion: official Hansard 39-1 (Apr 2006) → 45-1 (live).

The official XML archive begins at the 39th Parliament: 36-2/37-1/38-x URLs
redirect to a 404 error page (verified). So "modern corpus" = 2006→now, fetched
with our speech-level parser, plus the already-fetched 44-1 (2021–2025).

Stages (progress streamed to overnight.log):
  1 fetch    : fetch_hansard.py per session (~1 h at 0.7 s delay)
  2 concat   : hansard-2006-now.txt, chronological
  3 merges   : 1024 BPE merges trained on an era-sampled 1M-char slice → vocab 1280
  4 encode   : every session → tokens.bin (uint16, appended in order), round-trip verified
  5 report   : tokens-info.json + MORNING.md (Colab runbook)

Backs up the 44-1-era artifacts (merges-441.json, tokens-441.bin) before
overwriting, and moves the stale smoke checkpoint to attic/.

  --smoke : tiny end-to-end check (2 sittings, 64 merges, smoke-suffixed outputs)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HANSARD = ROOT / "hansard"

PRE_44 = [(39, 1), (39, 2), (40, 1), (40, 2), (40, 3), (41, 1), (41, 2),
          (42, 1), (43, 1), (43, 2)]
POST_44 = [(45, 1)]
CHRONO = PRE_44 + [(44, 1)] + POST_44          # 44-1 read from existing input.txt

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
args = ap.parse_args()

if args.smoke:
    NUM_MERGES, DELAY, MAX_SIT = 64, 0.5, 2
else:
    NUM_MERGES, DELAY, MAX_SIT = 1024, 0.7, 400

T0 = time.time()


def log(stage: str, msg: str) -> None:
    print(f"[{stage:7s}] {msg}   (+{time.time() - T0:,.0f}s)", flush=True)


# ------------------------------------------------------------------ stage 1 --
def fetch_sessions() -> None:
    targets = [(39, 1)] if args.smoke else PRE_44 + POST_44
    for parl, sess in targets:
        suffix = "smoke-" if args.smoke else ""
        out = HANSARD / f"input-{suffix}{parl}-{sess}.txt"
        min_size = 1_000 if args.smoke else 200_000   # smoke = 2 sittings only
        if out.exists() and out.stat().st_size > min_size:
            log("fetch", f"{parl}-{sess}: cached ({out.stat().st_size:,} B)")
            continue
        for attempt in (1, 2):
            log("fetch", f"{parl}-{sess} (attempt {attempt}) ...")
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, "fetch_hansard.py", "--parl", str(parl),
                 "--session", str(sess), "--lang", "E", "--out", out.name,
                 "--max-sittings", str(MAX_SIT), "--delay", str(DELAY)],
                cwd=HANSARD, capture_output=True, text=True)
            lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
            log("fetch", f"  {time.time() - t0:,.0f}s · " +
                (" | ".join(lines[-2:]) if lines else "(no output)"))
            if r.returncode != 0 and r.stderr:
                log("fetch", f"  rc={r.returncode} · " +
                    r.stderr.strip().splitlines()[-1][:200])
            if out.exists() and out.stat().st_size > min_size:
                break
        else:
            log("fetch", f"!! {parl}-{sess} FAILED — continuing without it")


# ------------------------------------------------------------------ stage 2 --
def concat_corpus():
    if args.smoke:
        src = HANSARD / "input-smoke-39-1.txt"
        dest = ROOT / "corpus-smoke.txt"
        text = src.read_text(encoding="utf-8")
        dest.write_text(text, encoding="utf-8")
        log("concat", f"smoke corpus: {len(text):,} chars")
        return dest
    dest = ROOT / "hansard-2006-now.txt"
    total = 0
    with open(dest, "w", encoding="utf-8") as w:
        for parl, sess in CHRONO:
            name = "input.txt" if (parl, sess) == (44, 1) else f"input-{parl}-{sess}.txt"
            src = HANSARD / name
            if not src.exists():
                log("concat", f"MISSING {name} — skipped")
                continue
            text = src.read_text(encoding="utf-8")
            w.write(text)
            if not text.endswith("\n"):
                w.write("\n")
            total += len(text)
            log("concat", f"{name}: {len(text):,} chars")
    log("concat", f"total {total:,} chars -> {dest.name}")
    return dest


# ------------------------------------------------------------------ stage 3 --
def sample_text(corpus: Path) -> str:
    if args.smoke:
        return corpus.read_text(encoding="utf-8")
    size = corpus.stat().st_size
    slices, chunk = 12, 100_000                 # era-diverse: 12 evenly spaced slices
    parts = []
    with open(corpus, "rb") as f:
        for k in range(slices):
            pos = int((k + 0.5) * (size - chunk * 4) / slices)
            f.seek(pos)
            parts.append(f.read(chunk * 4).decode("utf-8", errors="ignore")[:chunk])
    return "".join(parts)


def get_stats(ids: np.ndarray) -> dict:
    combo = ids[:-1].astype(np.int64) * 65536 + ids[1:]
    uniq, counts = np.unique(combo, return_counts=True)
    return {(int(c) >> 16, int(c) & 0xFFFF): int(n) for c, n in zip(uniq, counts)}


def greedy_keep(hit: np.ndarray) -> np.ndarray:
    keep, last = [], -2
    for i in hit:                               # 'aa' pairs can overlap ('aaa')
        if i > last + 1:
            keep.append(i)
            last = i
    return np.array(keep)


def apply_merge(ids: np.ndarray, pair, idx: int) -> np.ndarray:
    a, b = pair
    hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
    if hit.size == 0:
        return ids
    keep = greedy_keep(hit) if a == b else hit  # a≠b ⇒ matches can't touch
    ids = ids.copy()
    ids[keep] = idx
    return np.delete(ids, keep + 1)


def train_merges(text: str) -> list:
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    log("merges", f"training on {len(text):,} chars ({len(ids):,} bytes)")
    out, t0 = [], time.time()
    for i in range(NUM_MERGES):
        stats = get_stats(ids)
        if not stats:
            break
        (a, b), n = max(stats.items(), key=lambda kv: kv[1])
        ids = apply_merge(ids, (a, b), 256 + i)
        out.append([a, b, 256 + i])
        if i % 128 == 0 or i == NUM_MERGES - 1:
            log("merges", f"  {i:4d}/{NUM_MERGES} · best ({a},{b}) ×{n:,} · "
                          f"{len(ids):,} ids · {time.time() - t0:,.0f}s")
    return out


# ------------------------------------------------------------------ stage 4 --
def build_tokenizer(merges_list):
    mmap = {(a, b): idx for a, b, idx in merges_list}
    tb = {i: bytes([i]) for i in range(256)}
    for a, b, idx in merges_list:
        tb[idx] = tb[a] + tb[b]
    return mmap, tb


def encode_text(text: str, mmap) -> np.ndarray:
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in mmap.items():
        hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if hit.size == 0:
            continue
        keep = greedy_keep(hit) if a == b else hit
        ids[keep] = idx
        ids = np.delete(ids, keep + 1)
    return ids.astype(np.uint16)


def decode_ids(ids, tb) -> str:
    return b"".join(tb[int(i)] for i in ids).decode("utf-8", errors="replace")


def encode_all(corpus: Path, mmap, tb):
    name_bin = "tokens-smoke.bin" if args.smoke else "tokens.bin"
    tok_out = ROOT / name_bin
    tok_out.unlink(missing_ok=True)             # fresh start (restart-safe)
    sessions = ([("smoke", corpus)] if args.smoke else
                [(f"{p}-{s}", HANSARD / ("input.txt" if (p, s) == (44, 1)
                                         else f"input-{p}-{s}.txt")) for p, s in CHRONO])
    info, total_chars, total_toks = [], 0, 0
    for name, path in sessions:
        if not path.exists():
            log("encode", f"{name}: missing — skipped")
            continue
        text = path.read_text(encoding="utf-8")
        ids = encode_text(text, mmap)
        for frac in (0.25, 0.5, 0.75):          # round-trip: encode→decode==identity
            seg = text[int(len(text) * frac):int(len(text) * frac) + 3000]
            if seg and decode_ids(encode_text(seg, mmap), tb) != seg:
                raise SystemExit(f"ROUND-TRIP FAILED in {name}")
        with open(tok_out, "ab") as f:
            ids.tofile(f)
        total_chars += len(text)
        total_toks += len(ids)
        info.append({"session": name, "chars": len(text), "tokens": int(len(ids))})
        log("encode", f"{name}: {len(text):,} chars -> {len(ids):,} tokens "
                      f"({len(text) / len(ids):.2f} c/t)")
    log("encode", f"total {total_chars:,} chars -> {total_toks:,} tokens "
                  f"({total_chars / total_toks:.2f} c/t) -> {name_bin}")
    return tok_out, info, total_chars, total_toks


# ------------------------------------------------------------------ stage 5 --
def report(corpus, merges_list, tok_bin, info, total_chars, total_toks) -> None:
    vocab = 256 + len(merges_list)
    cpt = total_chars / total_toks
    payload = {"vocab": vocab, "merges": len(merges_list), "sessions": info,
               "total_chars": total_chars, "total_tokens": int(total_toks),
               "chars_per_token": round(cpt, 3), "tokens_file": tok_bin.name}
    name_info = "tokens-smoke-info.json" if args.smoke else "tokens-info.json"
    (ROOT / name_info).write_text(json.dumps(payload, indent=2))
    if args.smoke:
        log("done", f"SMOKE OK · vocab {vocab} · {total_toks:,} tokens · {cpt:.2f} c/t")
        return
    upload_mb = tok_bin.stat().st_size / 1e6
    (ROOT / "MORNING.md").write_text(f"""# Morning run — capstone on the 2006–now corpus

## Built overnight
- `hansard-2006-now.txt` — {total_chars:,} chars, {len(info)} sessions (39-1 → 45-1 + reused 44-1)
- `merges.json` — {len(merges_list)} merges, **vocab {vocab}** (trained on an era-sampled slice)
- `tokens.bin` — **{total_toks:,} tokens** (uint16, {upload_mb:,.0f} MB) at {cpt:.2f} chars/token
- `tokens-info.json` — per-session table
- Old 44-1 tokenizer backed up as `merges-441.json` / `tokens-441.bin`

## Colab (T4 runtime)
1. Upload three files from this folder: `gpt_tokens.py`, `merges.json`, `tokens.bin`
   ({upload_mb:,.0f} MB — the only big one)
2. Run:
```
!python gpt_tokens.py --preset default --sft --steps 12000 --batch 16
```
   - step-0 loss should read ≈ {np.log(vocab):.2f} (= ln {vocab})
   - 12,000 steps × 16 × 512 = ~96M tokens · ETA ~5–7 h on T4
3. Plan B (if tokens.bin upload stalls): upload `hansard-2006-now.txt` instead and
   add `--max-encode-chars 999999999` (re-encodes ~25 min on Colab before training)

## Bring back
Loss curve + the three `[prompt]/[model]` samples + SFT Q/A answers.
We then compute **bits per character** vs the char model's 1.378 nats/char —
the fair cross-tokenization verdict.
""")
    log("done", f"vocab {vocab} · {total_toks:,} tokens · {cpt:.2f} c/t · "
                f"MORNING.md written")


def main() -> None:
    fetch_sessions()
    corpus = concat_corpus()
    if not args.smoke:
        attic = ROOT / "attic"
        attic.mkdir(exist_ok=True)
        for src, dst in [("merges.json", "merges-441.json"),
                         ("tokens.bin", "tokens-441.bin"),
                         ("gpt_tokens.pt", "attic/gpt_tokens-smoke.pt")]:
            p = ROOT / src
            d = ROOT / dst
            if p.exists() and not d.exists():
                shutil.move(str(p), str(d))
                log("backup", f"{src} -> {dst}")
    merges_list = train_merges(sample_text(corpus))
    tok_json = ROOT / ("merges-smoke.json" if args.smoke else "merges.json")
    tok_json.write_text(json.dumps(merges_list))
    log("merges", f"saved {len(merges_list)} merges -> {tok_json.name}")
    mmap, tb = build_tokenizer(merges_list)
    tok_bin, info, total_chars, total_toks = encode_all(corpus, mmap, tb)
    report(corpus, merges_list, tok_bin, info, total_chars, total_toks)
    if args.smoke:
        corpus.unlink(missing_ok=True)
        (HANSARD / "input-smoke-39-1.txt").unlink(missing_ok=True)
        log("done", "smoke artifacts tidied")


if __name__ == "__main__":
    main()
