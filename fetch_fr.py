# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch French Hansard XML for every session 39-1 → 45-1 (mirrors the EN overnight).

44-1 FR already exists (input-fr.txt + cached -F XMLs) — skipped.
Includes the 42-1 lesson: max-sittings 450 for the longest parliament.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HANSARD = ROOT / "hansard"
SESSIONS = [(39, 1), (39, 2), (40, 1), (40, 2), (40, 3), (41, 1), (41, 2),
            (42, 1), (43, 1), (43, 2), (45, 1)]
T0 = time.time()


def log(m):
    print(f"[fetch-fr] {m}   (+{time.time() - T0:,.0f}s)", flush=True)


for parl, sess in SESSIONS:
    out = HANSARD / f"input-fr-{parl}-{sess}.txt"
    if out.exists() and out.stat().st_size > 200_000:
        log(f"{parl}-{sess}: cached ({out.stat().st_size:,} B)")
        continue
    for attempt in (1, 2):
        log(f"{parl}-{sess} (attempt {attempt}) ...")
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "fetch_hansard.py", "--parl", str(parl),
             "--session", str(sess), "--lang", "F", "--out", out.name,
             "--max-sittings", "450" if (parl, sess) == (42, 1) else "400",
             "--delay", "0.7"],
            cwd=HANSARD, capture_output=True, text=True)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        log("  " + (f"{time.time() - t0:,.0f}s · " +
                    (" | ".join(lines[-2:]) if lines else "(no output)")))
        if out.exists() and out.stat().st_size > 200_000:
            break
    else:
        log(f"!! {parl}-{sess} FAILED — continuing")
log("done")
