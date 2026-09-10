# GPT-0 Academy — Roadmap

*Written Sep 8, 2026, after the first overnight scale-up. This is the plan we agreed on — update it, don't re-litigate it.*

---

## 0. Where we stand tonight

**Owned assets (updated Sep 8, evening)**
- **Federal Hansard EN+FR:** 1.77B chars (`hansard-bilingual.txt`), speech-level, 39-1 → 45-1.
- **Historical Hansard EN (CC0):** `hansard/lm/century-pre2006.txt` — 3.04B chars, 11,818 sitting days, 1901–2005.
- **Federal statutes + regs EN+FR:** 287.7M chars (`statutes/text/`), cloned from `justicecanada/laws-lois-xml` (OGL-Canada).
- **SFT data:** bilingual QP pairs mined from XML (`sft/qp-pairs.jsonl` + FR; `sft-stream.bin` built).
- **Models:** char model (242K) · 989K · Colab 11.33M EN base · `gpt-11m-sft-bilingual-named.pt` (bilingual QP SFT). Next: pretrain v3 on the sovereign census.
- **Tokenizer:** 1,024 merges / vocab 1,280 (EN-era). v3 spec (Sep 9): custom byte-level BPE, 32k–40k vocab, on the deduped bilingual census sample — §2d.
- **Published:** `huggingface.co/datasets/NathanielArfin/canadian-hansard-2006-now` (258.2M-token stream, card, verified).
- **LEGAL.md:** per-source license/access audit (Crown/OGL/Speaker's permission/CC) — every source has a row.

**Fleet (Sep 8 evening): 13 fetchers running**
- committees ×3 (the swing variable) · gazette (HTML era) · TBS policies · Ontario e-Laws statutes · Manitoba laws · NB Hansard (PDF pipeline) · Alberta Hansard · StatCan Daily · internal reports (CKAN audit/eval) · committee gov-responses · Wayback: Quebec laws + Quebec journals
- **Done today:** Senate (409.7M) · Ontario CSVs (304MB) · Ontario Wayback (460.7M incl. volumes) · baber (3.04B chars) · statutes extraction (287.7M)

**Census on disk: ~2.2B tokens.** Converging band at exhaustion: ~125–225M params (section 2b).

**Measured so far (Colab 11.33M, mid-run):** val 2.194 nats/token @ step 13,250 ≈ **1.015 bpc** — char model (1.99 bpc) already halved. Scaling ladder on identical data: 989K → 11.33M gap ≈ 0.6 nats at equal steps.

---

## 0b. Tonight's ledger — Sep 8, 20:24 ET (for posterity)

**Diagnosis — "the way the answers are coming out."** The garbled QP answers
traced to SFT starvation, not a codebook bug: the shipped 11M SFT runs did
300 steps × 12 × 512 ≈ 1.84M tokens over a 36.5M-token stream = **0.05 epochs**
at a flat 1e-5 — the model learned the name-block pattern but never the Q→A
mapping. Proven by the `/tmp/opencode/probe_garble.py` loop (deterministic,
greedy+seeded, codebook + FR-density + distinct-2 metrics):

- codebook-mismatch hypothesis **refuted** — sft-stream.bin decodes cleanly
  (14 `Q:` markers) under the checkpoint's own EN merges; bilingual-merges
  decode is mojibake (stream was EN-merged, consistently)
- **EN-only SFT (`gpt-11m-sft-en.pt`) is the best 11M model** — clean register,
  zero code-switching, distinct-2 0.34–0.47; bilingual runs contaminate EN
  outputs (EN tokenizer byte-falls-back FR; 11M params can't learn FR at 0.05
  epochs) — language-locking must be per-checkpoint, not per-dataset-mix
- **attribution prefixes cost ~20 boilerplate tokens** before content at this
  scale; the "named" variant is the worst-behaved. Defer naming to ≥50M params
- `sft_hansard.py` **still carries the retracted target-shift bug** (labels not
  shifted +1; `gpt_tokens.py:375` has the correct pattern). Do not re-run it
  as-is; fix before the tiny-model path is used again

**Disk:** ~36 GB freed (uv/npm/bun caches incl. a half-extracted CUDA torch;
electoral-atlas research duplicate — history intact on
github.com/wra-sol/electoral-atlas; ~7.9 GB node_modules across 18 parked
projects; stale worktree entries pruned, `callcaster-hardening` worktree
removed with branch preserved). Box runs 4 cores; tonight it sat at load 11.5.

**SFT v3 in flight at this timestamp** — `sft_v3_en.py`: fresh from
`gpt-11m-base.pt`, EN-locked 35,401 mined pairs (12.23M-token stream), 3,000
steps ≈ 1.5 epochs (vs 0.05), warmup + cosine 1e-5→1e-6, replay/val gauges,
checkpoint + resume every 500 steps. Step 300+: task 1.69–1.82 (vs the old
run's flat 2.1–2.6), replay ~2.0 (no forgetting). ETA ~01:40 ET under
contention. Verdict to follow: probe + `eval_hansard.py` vs the old SFTs.

---

## 1. The module ladder (what comes next)

### Module 6 — The raw capstone (finishing now)
1. Colab run completes → bring back loss curve, 3 samples, SFT Q/A.
2. **Fair comparison:** bpc = nats/token ÷ 3.12 ÷ ln 2 vs char model's 1.99. Verdict on "did tokenization + capacity + data buy understanding."
3. **SFT v2:** bilingual, language-locked, on mined QP pairs (35K → ~60-70K with FR). Trainer patched: `--sft-data sft/qp-pairs.jsonl`, sentinel-packed stream (cached), all-masked-batch guard.
4. **Eval suite v1** (`eval_hansard.py`): held-out ppl/bpc · held-out QP Q/A (incl. "What is a filibuster?" — never in training) · register-start rate · distinct-2 repetition.
5. **DPO:** preference pairs (reject: off-register/off-topic generations; chosen: mined ground truth). ~40-line loss. After evals exist.
6. Publish checkpoint + model card next to the dataset on HF.

### Module 7 — Stand on the giants (LoRA)
- **Base: Tiny Aya 3.35B** (Cohere Labs, Toronto; 70+ languages; open license). Alternates: GLM-4-9B (MIT), Qwen (ungated); Aya Expanse 8B if we accept the gated-token dance.
- QLoRA on Colab T4 (Unsloth stack) or Kaggle 2×T4; Modal ≈ $1–4 if needed.
- Train on the SAME language-locked bilingual QP pairs. Same eval suite judges both models — same held-out pairs, same metrics.
- **Cross-lingual probe:** same questions in FR/ES/AR/HI/TL/SW, base vs adapted — measures the alignment tax and the transfer gradient. Register (the "Mr. Speaker" cadence) is EN/FR-only by construction; comprehension is the multilingual claim.
- **Demo: "Ask Parliament, in your language"** — one box, any language, closed-book answers from the House's own record.
- **Naming rule:** "Tiny Aya, Canadian-civic-tuned by us" — lineage always in the card. Never "a Canadian model" bare.

### Module 8 — The branch (continued pretraining)
- Take a 3–4B open base, **continue pretraining** on `hansard-bilingual` (+ early provincial/CanLII slices). ~$200–1,000 GPU.
- Claim earned: "Canadian-adapted." Real lineage, real training.

### Module 9 — The nano-lab (from scratch)
- 1–2B params × 20–40B tokens on the **Canadian civic mix**: bilingual Hansard (federal + provincial) + CanLII + legislation. ~$1,500–8,000 (8×H100, days); DRAC allocation could make it free.
- nanochat-style pipeline exists to copy (tokenizer → pretrain → SFT → chat).
- Only here does "pretrained in Canada" become true. **This is where provincial data finally becomes rational** — the Phase-3 data plan.
- Then post-train it ourselves: SFT v2 pairs + DPO. Full vertical integration.

**Canada's structural advantages:** Crown copyright (no licensing hell on gov works — proven by this corpus), DRAC free compute, Mila/Amii/Vector/CIFAR ecosystem, SR&ED, and a bilingual mandate that is a genuine gap in open weights.

---

## 2. Data workstreams

| Task | Status | Notes |
|---|---|---|
| EN corpus 2006→now | ✅ done | 806.2M chars, verified session-by-session |
| FR corpus fetch | ✅ done | all 12 sessions, ~966M chars (44-1 = `input-fr.txt`) |
| Committee Evidence EN+FR | 🔄 fetching | 3 background workers (`committees/fetch_all.py`); XML pattern verified Sep 8; per-meeting discovery (no bulk enumeration exists); resumable at XML cache |
| Dedup (paragraph ≥200 chars, exact) | script ready | dedupe.py → `hansard-bilingual.txt` (EN then FR per session) |
| Bilingual merge retrain | pending fetch | era-sampled slice, EN+FR mix; keep 1024 merges (comparability) |
| Re-encode tokens | pending | → tokens.bin v2; trainer auto-detects |
| Mine FR QP pairs | ✅ done | bilingual QP pairs mined (`sft-stream.bin` + labels) |
| **SFT v2 + post-training stack (user direction, Sep 8)** | design pending | SFT is being *modified* and followed by additional post-training stages. Inventory available: QP pairs (EN 35.4K + FR + held-out 722), committee Q&A (witness↔member exchanges, in flight), written questions (~11K long-form pairs, backlog), Senate QP. Post-stage options: domain-weighted annealing on highest-quality subset (Hansard+statutes), DPO/preference construction (register pairs, citation-grounding), rejection-sampling RAFT against corpus grounding, held-out eval by era/domain. **User will steer specifics.** |
| Written questions (`WrittenQuestionResponse`) | backlog | ~5/sitting ≈ +11K long-form factual pairs |
| Bilingual merge retrain | ✅ done | `hansard-bilingual.txt` (1.74B chars) |
| Historical Hansard (baber, CC0) | ✅ done | `hansard/lm/century-pre2006.txt` — 3.04B chars, 1901–2005 |
| Provincial corpora | **in progress** | ON ✓ (CSV+Wayback+e-Laws) · QC (Wayback, running) · MB ✓ · NB ✓ · AB ✓ · BC ✅ (CiviX REST API, 890+ acts, structured XML) · SK ✅ (Freelaw REST API, 508 acts + regs, PDF) · remaining: NS/NL/PE/YT/NT/NU |
| Tokenizer v3 + re-encode | pending fetches | custom byte-level BPE, 32k–40k vocab (spec §2d, node 2), trained on a 5GB **deduped** bilingual census sample; doc boundaries via multipack cu_seqlens at pack time, not sentinels |

**SFT language-locking:** data is pure per file (`-E` → en, `-F` → fr); uniform `Q:`/`A:` markers; the prompt's language selects the answer's language. No code-switching by construction.

---

## 2b. Corpus exhaustion ledger — tokens first, then params (Sep 8)

**The rule: params are set from the token census, not the inverse.** Measure every
exhaustible sovereign corpus, count unique tokens, then set params ≈ tokens ÷ 20
(Chinchilla). 2–3 epochs of repetition is the small-model knob afterward. The
model size is an output of the data campaign, not an input.

| Corpus | Status | Chars (EN+FR) | Tokens (est @ ~3.1 c/t) |
|---|---|---|---|
| Federal Debates (Hansard) EN+FR | ✅ measured | 1.77B | ~560M |
| Committee Evidence EN+FR | 🔄 fetching (3 workers) | est 3–6B | est 1.0–2.0B |
| Statutes + regulations EN+FR | ✅ measured + extracted — cloned `justicecanada/laws-lois-xml` (15,360 XMLs), tag-stripped to `statutes/text/{eng,fra}/` | 287.7M | **~93M** |
| **Ontario statutes + regs EN/FR** | 🔄 fetching — e-Laws v2 API cracked (autocomplete enumerates w/ per-language aliases; `doc-search/{alias}` returns full text; FR = genuine French). stdlib fetcher, no browser needed. King's Printer permission = reproduce without charge | est 0.4–1B chars | est 130–300M |
| Manitoba statutes + OICs EN/FR | 🔄 fetching — 516 consolidated statutes + orders section, curl-accessible; FR pages partial (bilingual subset), stubs auto-skipped | est 0.15–0.3B chars | est 50–100M |
| **NB Hansard EN/FR** | 🔄 fetching — PDFs per sitting day (592 across 9 session archives, 58th→61st assemblies), extracted with pdftotext; PDFs carry bilingual headers (EN/FR interleaved). Built on the new PDF-extraction pipeline | est 0.2–0.4B chars | est 60–130M |
| OER / open textbooks (multi-province) | ✅ **enumeration COMPLETE (Sep 8)** — eCampusOntario **🔄 running via API** (9,515 items; pressbooks+DSpace) · BC/YT: pressbooks.bccampus.ca + collection.bccampus.ca ✓ (browser) · AB: pressbooks.openeducationalberta.ca ✓ (browser) · SK: openpress.usask.ca ✓ (browser) · Atlantic: pressbooks.nscc.ca ✓ (browser) · MB program defunct (wayback for history) · QC needs alt names (profweb.ca next) · ecampusalberta.ca = squatter, ignore · ComputeOntario = curated micro-lane (~10 training texts) · AU Press ✓ (200, EN+FR) | est 0.5–1.5B chars | est 160–480M |
| **PDF-extraction pipeline** | ✅ built (pdftotext/poppler wired into the NB fetcher; generic extractor pattern ready) — unlocks: NB, NWT, curricula, gazette back-issues, OAG/PBO/CMHC/TSB reports, OIC PDFs | — | — |
| **Older federal Hansard (pre-2006)** | recon — official XML floor is 39-1 (Apr 2006, verified). Channels: **LIPAD (lipad.ca, structured Hansard 1901→, EN+FR, research-built)** — 403s curl AND agent fetch; headless-Chromium test pending, license verify on-site · **parl.canadiana.ca** (1867→ digitized scans; OCR exposure to verify) · archive.org scans · **License: Hansard published pre-~1975 is public domain in Canada (Crown = 50 yrs from publication) — strongest possible; post-1975 = Crown non-commercial** · 1901–2005 volume could 2–3× the federal Hansard corpus; older register suggests a separate capped slice | est 2–4B chars | est 0.7–1.3B |
| **nav.do decision-app family (Decisia platform)** | ✅ 3 confirmed members: **SST** (decisions.sst-tss.gc.ca), **SCT**, **NS Courts** — one parser likely covers the family + probable members across provincial courts/boards. **This is the highest-leverage single build in the adjudicative wave** | est 0.3–1B chars | est 100–350M |
| Labour/procurement lanes | CIRB one hop (404 on guess) · WCAT/WSIAT (workers' comp appeals) unprobed · Supply Manual/SACC: buyandsell.gc.ca DNS-dead → moved to canada.ca (browser bucket) · tender documents: provincial portals → PDF wave | est | est |
| Budget documents (federal + 13 jurisdictions + municipal) | wave 3 — 119 directory entries; budget-plan chapters = long policy prose, mostly PDF → PDF wave | est 0.2–0.5B chars | est 60–160M |
| School boards (47 directory entries, quantified) | wave 3 — heterogeneous portals; TDSB exemplar measured (policy PDFs ✓); framework crawl per board | est 0.3–1B chars | est 100–300M |
| **Provincial/territorial courts + tribunals (sweep measured)** | ✅ sweep done: 12/13 court systems curl-accessible (BC 000-refused, QC 403-blocked); BC LRB/BCSC ✓, AB securities ✓ (PDFs), OLT ✓, YT courts (5 PDFs) — each needs one-hop recon to its decisions DB, then bespoke fetchers (nav.do family parser where shared) | est 0.5–1.5B chars | est 150–500M |
| Provincial OICs (BC/AB/SK/QC/NS gazettes) | wave 3 — mostly provincial-gazette PDFs → PDF-extraction wave; Ontario's OICs/O.Reg already inside the e-Laws harvest; Manitoba OIC section riding the MB fetch | est | est |
| Quebec Journal des débats (commissions **+** chamber) | ✅ commissions fetching (12,706 transcripts, ~3B chars); **chamber postback CRACKED** — two-step (dropdown change postback → search) + Suivante walk; 103 sessions, ~4.5K days tracking (~1.4–2.3B chars); total lane ~4.5–5B chars FR-native | est 4–5B | est 1.3–1.7B |
| **BC statutes (CiviX REST API)** | 🔄 fetching — `bclaws.gov.bc.ca/civix/content/complete/statreg/` returns structured XML; 26 letter dirs → 890 act dirs → documents; XML-to-text extraction; King's Printer Licence (permissive); English only | est 0.2–0.5B chars | est 60–160M |
| **SK statutes + regs (Freelaw REST API)** | ✅ done — 1,253 docs, 77.6M chars, 0 errors; `publications.saskatchewan.ca/api/v1/freelaw/acts` returns full catalogue in one JSON call; PDFs via `/products/{id}/formats/{fid}/download`; ~10% have FR PDFs; Crown copyright (personal use only, redistribution prohibited) | 77.6M | ~25M |
| **SK Gazette (Freelaw API, 1958–2026)** | ✅ done — 3,185 issues, 446.1M chars, 2 errors (category 1511 → 69 years → Part I/II; thread-pooled free digital PDFs) | 446.1M | ~140M |
| **SK Orders in Council (Freelaw API, 1946–2026)** | 🔄 fetching — category 347 → 21 year cats → ~11K products (short reg orders, ~3-8K chars each); SK API intermittent (was HTTP 000) — enumeration retrying; same Crown personal-use terms | est 60–120M | est 20–40M |
| **NB statutes (Cyberlex PDFs)** | ✅ done — 421 statutes, 62.8M chars; `laws.gnb.ca/en/pdf/cs/{chapter}.pdf`; Crown copyright; English (FR blocked by site) | 62.8M | ~20M |
| **NS statutes + regs** | ✅ done — statutes 560/576 (18.7M, finalizing at Crawl-delay-10); regs 1,425/1,427 (22.0M, 2 errors) | ~41M | ~13M |
| **NL statutes (assembly.nl.ca)** | ✅ done — 403 docs, 19.9M chars (3 errors); titleindex.htm -> 406 codes; EN only | 19.9M | ~6M |
| **PEI legislation (princeedwardisland.ca)** | ✅ done — 790 docs (414 acts + 376 regs), 29.2M chars, 0 errors; enumeration via Wayback list snapshot, downloads from live (non-WAF'd) site | 29.2M | ~9M |
| **NT statutes + regs (justice.gov.nt.ca)** | ✅ done — 808 docs (239 acts + 569 regs), 77.0M chars, 0 errors; one A-Z page; bilingual EN/FR PDFs | 77.0M | ~24M |
| **NU consolidated law (nunavutlegislation.ca)** | ✅ done — 476 docs, 26.8M chars, 0 errors | 26.8M | ~8M |
| **YT legislation (laws.yukon.ca)** | ✅ done — 1,474 docs (432 acts, 1,042 regs), 166.1M chars, 415 errors (78% of CDX set; Wayback route) | 166.1M | ~53M |
| **Federal committee evidence (ourcommons.ca)** | 🔄 fetching — plan priority #2; static HTML→raw XML, EN+FR; 943 meetings in 45-1, ~7,500 across 43-1/44-1/45-1 (~15K XML files); sessions back to 35-1 (1994); robots-permitted — see DISCOVERY-N | est 1–2.5B chars | est 300–800M |
| **CRA publications (Folios + ITs + ICs + GST-HST)** | ✅ done — Folios 97 EN + 98 FR (7.0M); pubs 1,034 (32.0M, 964 stub-follows; ~316 legit short/absent); OGL v2.0 | 39M | ~13M |
| Ontario Hansard (House + committees) | 🔄 CSV era 2018→ downloaded via headless Chromium (browserfetch/ontario/csv/) · **2006–2018 harvesting from Wayback: 3,763 day snapshots, in flight** (volume-variant pass queued; chrome-strip refinement owed) · Speaker's permission recorded | est 1–3B | est 300–900M |
| Senate Debates (chamber) | ✅ **done** — 1,456 sittings (1,451 FR), 409.7M chars EN+FR | measured | ~130M |
| **Historical Hansard EN (baber, CC0)** | ✅ **done** — 11,818 sitting days, 1901–2005 (104 yrs with data), 3.04B chars → `hansard/lm/century-pre2006.txt`. Measured EN-only (633 diacritics/39.4M chars in 1990). **Mix-cap recommendation: 10–15% of pretrain tokens; downweight pre-1945** | 3.04B | **~970M** |
| Ontario Wayback (ola.org era) | ✅ first pass done — 3,763 day snapshots, 176.9M chars (chrome-strip refinement owed); volume-variant pass (hansard-1/-2) queued | measured so far | ~55M |
| **Alberta Hansard** | 🔄 built — docs.assembly.ab.ca PDFs per sitting (`legislature_{n}/session_{s}/{date}_{time}_han.pdf`), curl ✓, pdftotext pipeline; smoke running; EN (FR translations not listed) | est 0.2–0.5B chars | est 70–160M |
| BC Hansard | recon — index pages expose only nav links (JS content); browser/Wayback bucket | est 0.3–0.8B chars | est 100–250M |
| LEGISinfo bills | recon — per-bill XML ✓ (200) but bulk bill-list endpoint still unfound (2 guesses 404/302) | est 0.3–0.8B chars | est 100–250M |
| House Journals (federal) | recon — all guessed XML patterns 302→404; DocumentViewer path untested; procedural register, moderate value | est | est |
| Curricula (QC/AB/BC/ON + school boards) | wave 3 — education.gouv.qc.ca (FR-native ✓), alberta.ca programs ✓, curriculum.gov.bc.ca ✓ all curl-accessible; program documents mostly PDF → rides the PDF-extraction wave; school boards (300+) join the municipal-minutes wave | est | est 30–100M |
| **Internal reports (audit/evaluation, CKAN)** | 🔄 **running** — 1,110 audit + 535 evaluation datasets → PDF → pdftotext; EN/FR pairs with department names | est 0.3–1B chars | est 100–350M |
| **Alberta CKAN (open.alberta.ca, full corpus)** | 🔄 running — CKAN package walk, 4,000/31,681 pkgs, 16,039 PDFs ok, 900M chars, est ~7B full run (biggest single lane); catch-all: statutes' regs, ministry pubs, annual reports; en|fr resources | est 4–7B | est 1–2B |
| Committee gov-responses (CKAN HTML) | 🔄 **running** — 22 resources confirmed, full catalog via dataset; reports themselves ride ourcommons/Wayback | est | est |
| **Open-access university presses** | ✅ measured — uOttawa (bilingual) ✓, U Calgary (CC-BY-NC) ✓, AU Press ✓, BCcampus (browser-only), eCampusOntario (SPA); MB/Memorial TBD | est 0.1–0.4B chars | est 30–130M |
| Theses (Theses Canada/LAC) | **excluded for now** — author copyright + legacy license = grey; not publishable-clean downstream | — | — |
| Canada Gazette (Parts I–III) | 🔄 fetching — HTML era 2014→ in flight; pubgc weekly listing harvester ✅ 1,137/1,219 pages, 147.6M chars, 597K PDF links census'd; measured 272K chars/issue (Part I, EN+FR) | est 0.5–0.8B | est 160–260M |
| **Municipal bylaws (big cities first)** | recon queued — **enumeration READY: 134 council portals from awesome-canada** (agendas/minutes/bylaws per city); consolidated bylaw HTML/PDF; 404s on first guesses | est 0.1–0.4B chars | est 30–130M |
| **SCT (Specific Claims Tribunal) decisions** | ✅ path FOUND (awesome sweep 2: `decisions.sct-trp.ca/sct/en/nav.do`) — host 403s curl entirely → browser-bucket Decisia (grammar known) | est | est |
| **Energy/utilities regulators (RÉQ, OEB)** | 🔄 fetching — RÉQ ✅ recon (FR-native, ~2,200 decisions + dossier pièces 10K+); OEB ✅ recon (~539 major decisions, 213K chars avg); AUC skip (AJAX+login), BCUC skip (WAF) — see DISCOVERY-M | est 0.2–0.5B | est 60–160M |
| **CSE annual reports** | curl 200 page (67KB) — report links JS-rendered, recon continues | est small | est |
| **Nunavut Hansard EN↔Inuktitut (HF, CC-BY-4.0)** | ✅ confirmed on HF (EdinburghNLP/nunavut-hansard-plusplus, 1M–10M rows) — covers NU legislature + an Indigenous language pair; harvest queued | est | est |
| NS Hansard | recon — day pages are stubs (no transcript, no PDF links, no iframes); real transcript location unknown → Wayback recon queued | est 0.1–0.3B chars | est 30–100M |
| **Provincial courts (ON/BC/AB/SK/MB…)** | ⚠️ RESOLVED (search-first): **archives live on CanLII = excluded** — SK closed, AB = 7-day recent PDFs only, MB = recent KB PDFs (curl ✓); the deep lanes are ON (OCA Boolean DB + SCJ/OCJ pages, browser) and BC (ASP.NET judgments DB since 1990, browser) | est 30–120M chars | est 10–40M |
| **CAI (Québec access commission) decisions** | ✅ DIRECT: 410 decision PDFs on one listing page (2009→), `/uploads/pdfs/decisions-en-surveillance/*.pdf`, curl OK — buildable now | est small–30M | est |
| **Régie de l'énergie du Québec decisions (FR)** | ✅ done — 2,199 decisions, 82.7M chars, 7 errors; FR-native | 82.7M | ~26M |
| **Ontario Energy Board major decisions** | 🔄 fetching — 539 unique records via WebDrawer PDFs, ~213K chars avg (~115M total); EN bodies + some FR summaries | est 100–150M | est 30–50M |
| **SOQUIJ jugements.qc.ca** | the funnel for ALL Quebec decisions; content API curl-OK; **search API uncaptured** (SPA, 2 interaction attempts failed, no public scraper) — dedicated interactive session queued | est 0.5–2B chars | est 160–640M |
| PCO (Privy Council) publications + manuals | recon — one 404 on guessed path; the Black Book (Manual of Official Procedure) + OIC database (1990→, canada.ca) both real; browser bucket | est | est |
| Government news releases (canada.ca + dept communications) | 🔄 recon — RSS hub mapped (recent-items only); historical needs per-department listings or the LAC web archive; canada.ca times out via curl, reachable via agent fetch | est 0.2–0.6B | est 70–190M |
| **StatCan The Daily EN/FR** | ✅ pattern cracked (lettered article pages: `dq{YYMMDD}{a-f}-{eng|fra}.htm`; curl-accessible) — fetcher queued. Statistical-journalism register, bilingual | est 0.3–0.6B chars | est 100–200M |
| SCC / Federal Courts decisions | SCC search app live (200, JS); Federal Court decisions path 404 on first guess — browser/Wayback recon queued. Highest-register Crown text | est 0.1–0.3B chars | est 30–100M |
| open.canada.ca CKAN sweep | ✅ swept — metadata/tabular land (OICs = Gazette Part II HTML, already covered; "court decisions" datasets = CSV statistics, not text). Confirms: primary sources > portals for LM text | — | — |
| Bank of Canada publications + speeches | wave 2 — monetary policy reports, staff notes, speeches; Crown corp; high-quality economic prose | est 0.1–0.3B | est 30–100M |
| LAC Government of Canada Web Archive | wave 3 — archived federal websites, decades of sovereign prose; needs access recon | est ? | est ? |
| Municipal council agendas/minutes (100+ eScribe/CivicWeb portals) | wave 3 — effort-heavy per-portal, moderate prose value (minutes) | est 0.3–1B | est 100–300M |
| Historical archives (BAnQ numérique, Canadiana.ca) | wave 3, optional — pre-1920 public-domain FR/EN; breaks the 2006 era floor, so capped slice only | est ? | est ? |
| NCTR records (nctr.ca) | **excluded from training absent community guidance** — culturally sensitive archive; not a default corpus row | — | — |
| SCC/Federal Courts judgments, OAG/PBO/StatCan/agency annual reports | triage — PDF-heavy, last wave | est | est 0.15–0.4B tokens |
| Written Q&A | already inside the Debates stream; SFT mining backlog | — | — |

**Converging totals (Sep 8 evening, rate-based projection): ~3.3–4.1B tokens → ~165–205M params** (floor ~2.9B → ~145M if pending lanes don't land; ceiling ~4.5B → 225M if the wide-band lanes overshoot). On disk now: **~2.1B tokens**. Biggest in-flight/pending contributors: committees (~400M, swing), GOQ gazette ~446M, eCampusOntario books ~650M (wide band, browser wave), StatCan Daily ~68M, Decisia ~91M. Chinchilla tokens÷20; freeze when fetches land. Plan compute around **~180M params** (fits one 24GB GPU). All §2b token figures are estimates at 3.12 chars/token (vocab 1280); the v3 tokenizer (§2d) re-derives them before the freeze.

---

## 2c. Corpus mix priority (Sep 8, user directive)

**Prioritize long, coherent text, in this order:**
1. Federal + provincial statutes and regulations (EN/FR where they exist)
2. Hansard and committee evidence
3. Canada Gazette
4. publications.gc.ca reports and guides
5. TBS / Privy Council / departmental policy manuals
6. Official form instructions and program guides (what citizens actually ask about)

Tier 1 is the license-cleanest class in Canada: most provinces explicitly
permit reproduction of consolidated laws without permission (Ontario's King's
Printer states it verbatim); ON/QC/MB/NB laws are bilingual by law.

---

## 2d. Pretrain v3 spec — decision matrix (Sep 9)
- **Corpus cap policy (Sep 9):** harvest-to-cap then prune at freeze — QC JD commissions tracks ~3B chars (FR-native, 12,706 sittings ~237K each); Alberta CKAN est ~7B; in-flight lanes run to completion, dedupe/prune at freeze to fit ~5B-token budget; disk (24G free) is the only live constraint, Alberta CKAN is the pruning trigger.
- **v3b target (Sep 9): 500M @ 1.5 epochs, params = 1.5×D/20 capped at 500M.** Census now 18.33B chars ≈ 5.9B tokens @3.12 chars/token (plan's Sep 8 §2b figures are stale — >3× growth). Requirement: D_honest ≥ 6.7B for full 500M; 'reliable' floor ~5.5–6B. Freeze gate: mini-census + real dedupe with the v3 tokenizer → set params from measured honest D (6.7B→500M, 6.0B→450M, 5.5B→412M, 5.0B→375M, 4.5B→337M). Alberta CKAN (~2B tokens) is the swing variable; queued lanes (budgets/school boards/manuals) are the expansion knob.
- **v3b target refinement (Sep 9): 500M @ 1 epoch is ideal → D = 10B honest tokens.** Freeze gate becomes params = D_honest/20 (single epoch, never undertrained). Current bank = ~5.9B raw (18.33B chars ÷ 3.12); full fleet landing → ~7–8.5B honest → 350–425M. Full 500M needs the EXPANSION PORTFOLIO (~+2–4B: regulatory decision bodies, securities regulators, school boards, budgets, departmental manuals, municipal fleet) → outer band ~9.5–11B honest. Most likely honest outcome 400–450M @ 1 epoch unless portfolio lands high. 1.5-epoch fallback (params=1.5D/20) stands if D lands <6.7B.
- **QC municipal FR pivot (Sep 9):** user directive — dive deep in Quebec for French, municipal/sub-municipal = gold. DISCOVERY-O cracked Montréal ADI portal: city council + 19 boroughs, static HTML year-paged, direct PDFs (typeDoc=pv/da), ~15-30K FR PDFs back to ~2001, curl-OK no WAF. Ranks: 1 Montréal ADI, 2 BAPE (460 dossiers, huge FR reports), 3 OCPM consultations, 4 Longueuil static PVs. Skip: Laval/Gatineau/QC-cities (JS or low-coherence), donnees.montreal CKAN (no PV datasets).
- **QC municipal FR builds (Sep 9):** MTL-ADI ✅ running (5,119 docs pv+da, city+19 boroughs, ~745M chars tracking — fetcher fetch_montreal_adi.py); BAPE ✅ running (461 dossiers → 7,362 vault docs, ~300M+ tracking — fetch_bape.py); OCPM ✅ running (58 consultations → thousands of docs — fetch_ocpm.py); Longueuil ✅ running (196 PV/ODJ — fetch_longueuil.py). Deep French municipal/sub-municipal seam flowing.
