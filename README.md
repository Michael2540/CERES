# CERES

> **Mission control for the first farm off Earth.** Built for the **Meta Global AI Developer Hackathon** (building period: Oct 27 – Nov 4, 2026).

**Repository created:** 2026-10-26. All code in this repository was written during the hackathon. See [HACKATHON.md](HACKATHON.md) for the build log and how to check that.

## Problem Statement

- [ ] **Build for Your World**: bring Meta's models into a domain we know firsthand.
- [x] **Push the Frontier**: use Meta's models on challenges in software, robotics, manufacturing, energy, materials, and beyond.

## Overview

Fresh crops will supplement astronaut diets on long missions, but crew time is scarce. On Mars, a question to Earth takes minutes each way. CERES is a crop-health copilot for a spacecraft greenhouse:

- **Measures** every tray with **SAM 3.1** (canopy and per-plant masks). The numbers (area, greenness, growth) are computed by code, not guessed.
- **Detects** with deterministic rules (wilting risk, growth lag, low VPD, sensor faults). An LLM never raises an alert.
- **Investigates** each alert with a **Muse Spark** "Flight Agronomist". It calls tools, compares against the control tray, looks at frames, and returns ranked probable causes. Every piece of evidence it cites is machine-checked against what its tools actually returned.
- **Proposes, never actuates.** Recommended actions wait for a human to approve them.

### What is real and what is simulated

CERES runs entirely in software, without hardware:

| Part | Status |
|---|---|
| Plant images | **Real** frames from the [Aberystwyth Leaf Evaluation Dataset](https://doi.org/10.5281/zenodo.168158): a phenotyping greenhouse, 4 trays of *Arabidopsis thaliana*, 36 days, CC BY 4.0 |
| Faults on trays B and C | **Simulated.** Tray B's frames are transformed to show wilting and tray C's are time-warped to show light-limited growth, from each tray's fault start. Labelled SIMULATED FAULT everywhere. |
| SAM 3.1 measurements | **Real** model output on those frames. The measurement is real; the input fault on B and C is simulated. |
| Sensors (temperature, RH, CO₂, soil moisture) | **Simulated** by a deterministic sensor hub, including an injected RH-sensor flatline |
| Agent assessments | **Real** Muse Spark calls. The agent is deliberately not told which inputs are simulated, so it is tested blind. |
| Mars light delay | **Simulated** (Mars mode, in progress) |

## Meta Models Used

| Model | Access | Role in CERES |
|---|---|---|
| `sam-3.1` | Meta Model API | Segments every tray frame into per-plant masks (text prompt "plant"). The masks feed canopy area, greenness and growth metrics. 844 frames segmented, cached so nothing is paid for twice. |
| `muse-spark-1.3` | Meta Model API | **Flight Agronomist.** A tool-calling agent that investigates alerts across sensor series, canopy metrics and tray images, and returns a strict-JSON assessment. Its citations are validated and it gets one retry. |
| `muse-spark-1.3-contributor` | Meta Model API | Development and evaluation runs at lower cost |
| Muse Code (`muse` CLI) | Meta's coding agent | Wrote the ALED dataset fetcher (`scripts/fetch_dataset.py`) and reviewed milestones. See [MUSE_CODE.md](MUSE_CODE.md). |

Planned and switchable: the Mars-mode edge model (`EDGE_MODE=recorded|remote_glimmer|spark_contributor`). Muse Glimmer 30B cannot run on our development laptop, so any edge run will be labelled with the model that actually produced it.

## Built During the Hackathon

| Feature | Status |
|---|---|
| Virtual camera: range-request sampler for the 61.6 GB ALED archive (downloads only the frames used) | ✅ |
| Simulated faults on real frames (wilting transform, low-light time-warp), labelled throughout | ✅ |
| SAM 3.1 client: stream parsing, raw-output cache, per-image spend ledger, budget guard | ✅ |
| Simulated sensor hub (same protocol as a real one) with fault injection | ✅ |
| Deterministic rules: WILT_RISK, GROWTH_LAG, LOW_VPD_RISK, SENSOR_FAULT (with suppression) | ✅ |
| Replayable mission clock; alert episodes; no data from after the current mission time ever reaches the UI | ✅ |
| Muse Spark Flight Agronomist with 8 logged tools, citation validator, SSE streaming | ✅ |
| Mission-control dashboard (Next.js) with provenance badges, charts, frame scrubber, agent panel | ✅ |
| Validation of SAM masks against ALED hand annotations | ⏳ M6 |
| Evaluation harness (≥ 8 scenarios, small n, reported as fractions) | ⏳ M7 |
| Mars mode: link delay, on-board triage, Earth-side audit | ⏳ M8 |

## Demo

- **Demo video (3 min):** _link coming soon_
- **Live product:** runs locally (see below)

## Getting Started

Requires Python 3.11+ and Node 20+.

```bash
git clone https://github.com/Michael2540/CERES.git
cd CERES
cp .env.example .env                 # add META_API_KEY
python -m venv .venv && .venv/Scripts/python -m pip install -e .[dev]   # use .venv/bin/python on macOS/Linux
```

Build the demo database:

```bash
python scripts/fetch_dataset.py --with-masks       # free; ~170 MB of sampled ALED frames into data/raw/
python scripts/segment_all.py --dry-run            # prints the SAM cost estimate (~$2.10 for 844 frames)
python scripts/segment_all.py --backend api        # paid: sam-3.1; results cached in .sam_cache/
python scripts/backfill_pots.py                    # assigns plants to pots (no model calls)
python scripts/replay.py --backend none            # simulated sensors + hour-by-hour rule replay
```

Run it:

```bash
make dev                         # backend on http://localhost:8000
cd web && npm install && npm run dev    # dashboard on http://localhost:3000
```

"Investigate with Muse Spark" in the dashboard makes one live, paid call to `muse-spark-1.3`. It costs about $0.10–0.20.

Tests: `make test` (backend), `make lint`, `cd web && npx playwright test`.

## Limitations

- ALED has no real stress treatments, so every fault is simulated on real images, and the evaluation can only measure how well CERES recovers *simulated* faults.
- ALED removes plants for destructive sampling every few days. CERES tracks growth per pot so these removals don't read as shrinkage.
- Thresholds are author-chosen and mostly relative to the control tray. There are no sourced Arabidopsis setpoints.
- Canopy area in cm² uses a nominal scale (the dataset has no calibration marker). Pixel area is the primary metric.

## Team

| Name | GitHub | Role |
|---|---|---|
| Michael Yan | [@Michael2540](https://github.com/Michael2540) | Everything |

## Acknowledgements and License

CERES is [MIT](LICENSE) licensed. Plant images: Bell, J. & Dee, H. M. (2016), *Aberystwyth Leaf Evaluation Dataset*, [doi:10.5281/zenodo.168158](https://doi.org/10.5281/zenodo.168158), CC BY 4.0. They are adapted (subsampled, downscaled, and partly transformed to simulate faults). See [NOTICE](NOTICE) for full attributions and third-party terms.
