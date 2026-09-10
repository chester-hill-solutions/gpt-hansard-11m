# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Dedup + assemble the bilingual corpus.

Per session (chronological): EN text then FR text. Paragraphs ≥ 200 chars are
globally deduplicated (exact match after whitespace normalization); short
lines (Mr. Speaker, headings, votes) always pass. FR documents are parallel,
not line-aligned — dedup runs independently per paragraph, which is correct.

Output: hansard-bilingual.txt + dedup stats.
"""
import hashlib
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
H = ROOT / "hansard"
CHRONO = [(39, 1), (39, 2), (40, 1), (40, 2), (40, 3), (41, 1), (41, 2),
          (42, 1), (43, 1), (43, 2), (44, 1), (45, 1)]
MIN_PARA = 200

RE_WS = re.compile(r"[ \t]+")
T0 = time.time()
seen = set()
kept_chars = dropped = 0


def log(m):
    print(f"[dedup] {m}   (+{time.time() - T0:,.0f}s)", flush=True)


with open(ROOT / "hansard-bilingual.txt", "w", encoding="utf-8") as w:
    for parl, sess in CHRONO:
        for lang, name in (
            ("en", "input.txt" if (parl, sess) == (44, 1) else f"input-{parl}-{sess}.txt"),
            ("fr", "input-fr.txt" if (parl, sess) == (44, 1) else f"input-fr-{parl}-{sess}.txt"),
        ):
            src = H / name
            if not src.exists():
                log(f"MISSING {name} — skipped")
                continue
            text = src.read_text(encoding="utf-8")
            out_lines = []
            for line in text.split("\n"):
                p = RE_WS.sub(" ", line).strip()
                if not p:
                    continue
                if len(p) >= MIN_PARA:
                    h = hashlib.md5(p.encode()).hexdigest()
                    if h in seen:
                        dropped += 1
                        continue
                    seen.add(h)
                out_lines.append(p)
            w.write("\n".join(out_lines) + "\n")
            n_chars = sum(len(l) + 1 for l in out_lines)
            kept_chars += n_chars
            log(f"{lang} {parl}-{sess}: {len(text):,} -> {n_chars:,} chars "
                f"({100 * n_chars / max(1, len(text)):.1f}%)")

log(f"done: {kept_chars:,} chars kept, {dropped:,} duplicate paragraphs dropped")
