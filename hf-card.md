---
license: other
license_name: crown-copyright-canada
license_link: https://www.canada.ca/en/transparency/terms.html
language:
- en
task_categories:
- text-generation
pretty_name: Canadian Hansard 2006–2026 (speech-level, tokenized)
size_categories:
- 100B<n<1T
---

# Canadian Hansard, 2006 → now (Parliaments 39-1 → 45-1)

Speech-level English Debates (Hansard) of the House of Commons of Canada,
parsed from the official XML archive at ourcommons.ca — plus a pre-tokenized
BPE stream ready for language-model training.

Built as an independent research corpus — official parliamentary record, published openly.

## Contents

| file | what it is |
|---|---|
| `hansard-2006-now.txt` | raw text, chronological, 12 sessions, 806,202,795 chars |
| `tokens.bin` | same text pre-tokenized: 258,174,079 tokens, raw little-endian **uint16** |
| `merges.json` | 1,024 byte-pair-encoding merges → vocab of 1,280 |
| `tokens-info.json` | per-session character/token counts |

## Coverage

| Session | Period | Sittings | Chars | Tokens |
|---|---|---|---|---|
| 39-1 | 2006–2007 | 175 | 61.0M | 19.4M |
| 39-2 | 2007–2008 | 117 | 40.3M | 12.8M |
| 40-1 | 2008–2009 | 13 | 3.4M | 1.1M |
| 40-2 | 2009 | 128 | 43.9M | 14.0M |
| 40-3 | 2010–2011 | 149 | 51.0M | 16.3M |
| 41-1 | 2011–2013 | 272 | 100.1M | 31.9M |
| 41-2 | 2013–2015 | 235 | 83.5M | 26.7M |
| 42-1 | 2015–2019 | 438 | 158.8M | 50.9M |
| 43-1 | 2019–2020 | 45 | 14.7M | 4.7M |
| 43-2 | 2020–2021 | 124 | 47.8M | 15.4M |
| 44-1 | 2021–2025 | 391 | 151.2M | 48.5M |
| 45-1 | 2025– (in progress) | 139 | 50.6M | 16.4M |
| **Total** | | **2,226** | **806.2M** | **258.2M** |

Every session compresses at 3.09–3.17 chars/token — the tokenizer generalizes
uniformly across two decades of political language.

## Loading the tokens

```python
import numpy as np
ids = np.fromfile("tokens.bin", dtype=np.uint16)   # 258,174,079 tokens
```

IDs `0–255` are raw UTF-8 bytes; `256–1279` are merged tokens (see
`merges.json`). `tokens.bin` and `hansard-2006-now.txt` are positionally
aligned: token `i` of the stream decodes into the same span of text.

## Tokenizer

1,024 byte-pair merges trained on a 1.2M-char **era-sampled** slice (12
evenly-spaced chunks across the 20-year corpus). Round-trip
(encode → decode → identical text) verified per session.

## Quirks worth knowing

- **40-1 has 13 sittings**: the 2008 coalition-crisis prorogation — the
  shortest modern session. The corpus *is* the constitutional crisis.
- **43-1 has 45 sittings**: COVID suspension (Mar 13, 2020) then the
  Aug 2020 prorogation. Confirmed complete — the archive truly ends there.
- **42-1 is the longest** (438 documents). Its archive extends past a naive
  400-sitting cap; a capped fetch silently loses its tail.

## Source & rights

Official Debates (Hansard), House of Commons of Canada,
`ourcommons.ca/Content/House/{parl}{sess}/Debates/...` XML, parliaments 39-1
(April 2006) through 45-1 (fetched September 2026). The official XML archive
begins at 39-1; earlier parliaments are not served at this endpoint.

© His Majesty the King in Right of Canada. Crown copyright. This dataset is
an independent research artifact and is not affiliated with or endorsed by
the House of Commons.
