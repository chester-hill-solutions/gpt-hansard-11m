# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Mine real Question Period exchanges from the cached Hansard XML.

In the modern schema every QP topic is its own <SubjectOfBusiness>, and
interventions carry Type="Question" / Type="Answer" — ground-truth labels,
no heuristics. Each Answer is paired with the most recent Question in the
same subject block (handles supplementary exchanges).

Output:
  sft/qp-pairs.jsonl     — training pairs (~98%)
  sft/qp-heldout.jsonl   — every 50th new pair, never trained on (~2%)
"""
import glob
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
XML = ROOT / "hansard" / "xml"
OUT = ROOT / "sft"
OUT.mkdir(exist_ok=True)

RE_SUB = re.compile(r'<SubjectOfBusiness\b.*?</SubjectOfBusiness>', re.S)
RE_TITLE = re.compile(r'<SubjectOfBusinessTitle>([^<]*)')
RE_BLOCK = re.compile(r'<Intervention Type="(Question|Answer)".*?</Intervention>', re.S)
RE_SPEAKER = re.compile(r'<Affiliation[^>]*>([^<]+)</Affiliation>')
RE_PARA = re.compile(r'<ParaText[^>]*>(.*?)</ParaText>', re.S)
RE_TAG = re.compile(r'<[^>]+>')
ENT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'",
       "&#160;": " ", "&nbsp;": " "}


def clean(s: str) -> str:
    s = RE_TAG.sub('', s)
    for k, v in ENT.items():
        s = s.replace(k, v)
    return re.sub(r'\s+', ' ', s).strip()


seen, pairs, heldout = set(), [], []
files = sorted(glob.glob(str(XML / "*" / "*-E.xml")) +
               glob.glob(str(XML / "*" / "*-F.xml")))
t0 = time.time()
print(f"{len(files)} sittings to mine", flush=True)

for n_files, f in enumerate(files, 1):
    try:
        xml = open(f, encoding="utf-8", errors="ignore").read()
    except OSError:
        continue
    rel = "/".join(Path(f).parts[-2:])
    for sub in RE_SUB.finditer(xml):
        block = sub.group(0)
        tm = RE_TITLE.search(block)
        topic = clean(tm.group(1)) if tm else ""
        cur_q = None
        for iv in RE_BLOCK.finditer(block):
            body = iv.group(0)
            sp = RE_SPEAKER.search(body)
            text = " ".join(clean(p) for p in RE_PARA.findall(body))
            typ = iv.group(1)
            if typ == "Question":
                cur_q = (sp.group(1) if sp else "", text)
            elif typ == "Answer" and cur_q:
                q_who, q_text = cur_q
                cur_q = None
                if not (120 <= len(q_text) <= 1500 and 100 <= len(text) <= 2000):
                    continue
                key = hashlib.md5((q_text[:150] + text[:150]).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                row = {"q": q_text, "a": text, "who_q": q_who,
                       "who_a": sp.group(1) if sp else "",
                       "lang": "fr" if f.endswith("-F.xml") else "en",
                       "topic": topic, "src": rel}
                (heldout if len(seen) % 50 == 0 else pairs).append(row)
    if n_files % 250 == 0:
        print(f"  {n_files}/{len(files)} sittings · {len(pairs):,} pairs "
              f"(+{time.time() - t0:,.0f}s)", flush=True)

with open(OUT / "qp-pairs.jsonl", "w") as w:
    for r in pairs:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "qp-heldout.jsonl", "w") as w:
    for r in heldout:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"DONE: {len(pairs):,} train + {len(heldout):,} held-out pairs "
      f"from {len(files)} sittings · {time.time() - t0:,.0f}s", flush=True)
