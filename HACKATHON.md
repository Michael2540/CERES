# Hackathon Submission Info

**Event:** Meta Global AI Developer Hackathon
**Repository created:** 2026-10-26
**Building period:** 2026-10-27 to 2026-11-04
**Submission deadline:** 2026-11-04, 9:00 PM ET

## Original Work Statement

CERES was created from scratch for this hackathon. No code, data, or assets from earlier projects are included. The planning document (PLAN.md) was drafted before the building period; it contains no product code. Tools mentioned in it as earlier options (e.g. Graphloop) are not used and are not dependencies. The git history starts on 2026-10-26 with an empty scaffold (this file, README, LICENSE). Every later commit is hackathon work.

All third-party libraries, data, and assets used are listed below with their licenses, so it is clear we have the rights to use them.

## Third-Party Dependencies & Assets

| Name | License | Usage |
|---|---|---|
| Aberystwyth Leaf Evaluation Dataset (ALED), Aberystwyth University / NPPC, Zenodo 10.5281/zenodo.168158 | CC BY 4.0 | Plant time-lapse frames and ground-truth masks (sampled; only a small subset is committed in `data/demo/`) |
| meta-sam-parser | SAM License | Parses `sam-3.1` segmentation output (installed dependency, not vendored) |
| certifi | MPL-2.0 | CA bundle for TLS to the Meta Model API |
| FastAPI, Pydantic, PyYAML, jsonschema, remotezip | MIT | Backend API, config, schema validation, dataset range reads |
| Uvicorn, httpx, NumPy, SciPy, python-dotenv, sse-starlette | BSD-3-Clause | Server, HTTP, numerics, env loading, SSE |
| OpenAI Python SDK, opencv-python-headless, python-multipart | Apache 2.0 | Meta Model API client (OpenAI-compatible), image ops, uploads |
| Pillow | MIT-CMU (HPND) | Image I/O |
| Next.js, React, Tailwind CSS, Recharts | MIT | Dashboard |
| Playwright | Apache 2.0 | UI smoke tests |

## Build Log

| Date | Work completed |
|---|---|
| 2026-10-26 | Created the public repository; added README, LICENSE, and hackathon info. |
| 2026-10-26 | Installed Meta Muse Code CLI (v1.4.13); added MUSE_CODE.md usage guide and CLAUDE.md. |
| 2026-10-26 | Ran `muse init` to generate AGENTS.md (Muse Code project rules). |
| 2026-10-27 | PLAN.md §0: software-only amendment (ALED dataset replaces camera and trays; simulated sensor hub; labeled simulated faults; $150 Meta budget cap). |
| 2026-10-27 | **Muse Code** (`muse-spark-1.3-contributor`, 17 calls) wrote `scripts/fetch_dataset.py` + tests (ALED range-request sampler). Muse Code also did a read-only review of M0. |
| 2026-10-27 | Spikes S1 (`sam-3.1`: 20/20 plants segmented on a real tray) and S2 (`muse-spark-1.3` tool calls, strict JSON schema, 6-image messages). Results in SPIKES.md. |
| 2026-10-27 | M0 scaffold: package layout, SQLite schema with spend ledger, API.md + fixtures, Next.js shell, CI. |
| 2026-10-27 | M1–M3: frame ingest, simulated faults on trays B/C, SAM client with cache and budget guard, canopy metrics, simulated sensor hub, rules. |
| 2026-10-28 | **`sam-3.1`** segmented all 844 frames ($1.95, 0 errors). M3b: pipeline, mission replay clock, read API, alert episodes. |
| 2026-10-28 | M4: **`muse-spark-1.3`** Flight Agronomist (tools, citation validator, SSE). Live: B wilt alert → valid assessment, underwatering 0.80, $0.094. |
| 2026-10-31 | Agent prompt: general sensor-fault signatures (a disagreeing sensor in another place is not a fault). Eval 11/11 in three runs (low x2, high). Eval spend total $9.13. |
| 2026-10-31 | Perception fixes (SAM colour trim, GT Dice 0.94) and M7 eval harness: 11 scenarios on **`muse-spark-1.3`** at low and high effort, 10/11 pass; a first run at 6/9 was traced to simulation flaws, fixed and kept on record. Eval spend $5.97. |
| 2026-10-30 | M6: **`sam-3.1`** validated against 55 ALED hand-annotated frames (39 new calls, $0.10) with a colour-threshold baseline; `/eval/area-validation` and the /trust card. |
| 2026-10-30 | M5: mission-control dashboard. Adversarial audit (honesty/compliance and end-to-end correctness) and fix round: per-pot tracking, latched wilt alert, validator and disclosure fixes. sam-3.1 re-segmented 83 tray B frames ($0.21). |

## Submission Checklist

- [x] Public repository
- [x] Open-source license (MIT)
- [x] Problem statement selected in README (Push the Frontier)
- [x] Meta models used and documented in README
- [x] Hackathon-built features listed in README (update as M6–M9 land)
- [ ] 3-minute demo video (hackathon-built features only), publicly accessible
- [ ] All team members (max 4) listed on the submission page
- [ ] Meta Model API credits claimed by each member (by Nov 4, 9 PM ET)
- [ ] Submitted on the CV platform by Nov 4, 9:00 PM ET
