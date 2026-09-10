# LEGAL.md — corpus compliance record

*Established Sep 8, 2026. This file is the license and access audit for every
source in the sovereign corpus. No source enters the pretrain mix without a
row here. Update it, don't assume it.*

---

## Summary verdict

Every source currently in the corpus is **Crown copyright (federal or
provincial) or Open Government Licence - Canada**, published by a Canadian
government as an official public record. Non-commercial research use with
attribution is permitted for all of them. Three conditions and three flags
apply (below). No source in the corpus is private, journalistic, or
third-party-licensed content.

## The three attribution conditions (canada.ca terms, verbatim)

Reproduction for non-commercial purposes is permitted **provided you**:

1. exercise due diligence in ensuring the accuracy of the materials reproduced
2. indicate both the complete title of the materials reproduced, as well as
   the author (where available)
3. indicate that the reproduction is a copy of the version available at the
   original URL

→ Applied to this project: every published artifact (dataset card, model
card, code) must carry per-corpus source titles, the originating institution,
and the source URL pattern. The existing `hf-card.md` pattern satisfies this
in shape; every new corpus ships with its own card before publication.

## Per-source register

| Source | License | Access rules (robots.txt, checked Sep 8) | Gaps / flags |
|---|---|---|---|
| House Debates (Hansard) EN+FR, ourcommons.ca | Crown copyright (federal); historical "Speaker's permission" tradition for Debates; non-commercial reproduction with attribution | `/Content/`, `/DocumentViewer/` NOT disallowed; only search/embed paths are | None. Dataset already published under `license_name: crown-copyright-canada` |
| Committee Evidence EN+FR, ourcommons.ca | Crown copyright (federal) | Same host, same rules | None |
| Statutes + regulations, justicecanada/laws-lois-xml | **Open Government Licence - Canada** (LICENSE.md in repo) — permits ANY use incl. commercial, with attribution | robots.txt: none. Zero requests: harvested by git clone | Cleanest license on the board |
| Senate Debates EN+FR, sencanada.ca | Crown copyright (Senate) | robots.txt: none present | Cite Senate "Important Notices" terms in the Senate dataset card before publishing |
| Canada Gazette Parts I–III, gazette.gc.ca | Crown copyright (federal) | robots.txt: none present | **Third-party content flag:** Gazette notices include submissions by private parties (public notices, applications). Crown copyright is not assured for every notice. Mitigation: card states mixed provenance; exclude nothing, but document it |
| Quebec Journal des débats, assnat.qc.ca | Crown copyright (Québec) | robots.txt `*` disallows ONLY the RSS pages and the Journal search page (`index-jd/recherche.html`). Journal content pages are NOT disallowed | **Enumeration constraint:** do not scrape via the disallowed search page; enumerate via per-commission/per-session listing pages or by POST-free means. AssNat terms page wording to be cited verbatim before publishing |
| Ontario Hansard, ola.org | **Speaker's permission (terms read Sep 8):** "display, print, reproduce, and otherwise use excerpts ... without charge" for uses that are "reasonable, fair and non-commercial, and must credit the Assembly"; subject to Canadian IP law and Assembly parliamentary privilege; responsible for third-party IP; no coat of arms/trademarks without the Speaker's permission | Site blocks programmatic clients (curl AND python urllib → 403; agent-side fetch passes). Harvest strategy: prefer OLA's own published Data-resources files (intended distribution); else agent-side batching | **UN-PAUSED with conditions.** Credit "Legislative Assembly of Ontario" in every artifact. Non-commercial only |
| Gov news releases, canada.ca | Crown copyright (federal) | canada.ca robots.txt: none shown; site reachable via agent fetch (curl times out) | Harvest after the access question settles; releases are Crown content |
| CBC / press journalism | **NOT government content — excluded by policy** | — | Keeps the "sovereign record" claim true. Recorded in PLAN.md decisions log |
| TVO / TFO (provincial broadcasters) | **EXCLUDED by policy.** TVO is a Crown organization but asserts its own copyright ("© The Ontario Educational Communications Authority") — broadcaster/charity, not legislative Crown; catalogue includes third-party producers' documentaries and journalism (same class as CBC). **Rule: Crown org ≠ Crown copyright ≠ open license — the license on the actual work governs.** Exception only if an explicitly CC-licensed collection surfaces | — | Consistent with the CBC line |
| Open textbooks (BCcampus, eCampusOntario OER) | ✅ **IN — explicit open licenses.** Government-FUNDED, not government-WORKS: the CC license governs. BCcampus sitewide CC-BY 4.0 (verified via browser; curl 403), individual texts CC-BY/CC-BY-NC/CC-BY-SA | Browser/pressbooks harvest; wave 3 | Long coherent prose, incl. FR from Ontario French colleges |

## Downstream rules

- **Now (research, non-commercial):** permitted for all sources above, with
  attribution. Training is research; distributing the corpus + checkpoints
  with cards is reproduction under the non-commercial permission.
- **If anything ever goes commercial:** Crown non-commercial permission
  converts to "prior written permission required" — contact each institution
  (OGL-Canada sources excepted: statutes are fine either way, attribution
  still required). Plan this as a licensing project BEFORE any monetization.
- **Fair dealing (Copyright Act s.29)** independently supports research use,
  but the permission above is what the claim rests on — keep it that way.
- **Official symbols:** the Canada wordmark, Arms of Canada, flags may not be
  reproduced even non-commercially — none of our text artifacts include them;
  do not put official symbols in branding/artwork for the models.
- **No Canadian database right** exists (no EU-style sui generis right) — the
  corpus as a compilation carries no extra layer.

## Hygiene rules already in force

- Polite rates only (0.4–0.6s delays; well under any published limit).
- Network errors retry; only genuine 404s count as misses (silent-truncation
  rule from the 42-1 lesson).
- Every fetched artifact is cached and re-derivable; provenance headers are
  part of the text format (`=== Source — date ===`).
- All findings recorded with dates; re-check robots.txt before any new
  enumeration path.
