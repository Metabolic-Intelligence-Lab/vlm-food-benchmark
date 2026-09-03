# VLM Food Benchmark

Benchmark dataset and full evaluation pipeline for **food image recognition and
portion size estimation with Vision-Language Models (VLMs)**, accompanying the
paper *"Benchmarking Open-Source Vision-Language Models for Food Image
Recognition and Portion Size Estimation"* (Marchetti, Capezzone, Esposito,
De Spirito, Maulucci — under review).

## Dataset

- **200 real-world meal images** (`GT_Mensa_2025/resized_images/001.jpg` …
  `200.jpg`, longest side ≤ 748 px, EXIF-stripped) of regular dishes served at
  a university canteen, photographed by six participants with their personal
  smartphones under free acquisition conditions (mostly top-down).
- **870 ground-truth food items** (314 unique descriptions, 1–7 items per
  image): item name + portion weight in grams. Every item was **weighed with a
  kitchen scale** at collection time (tare of the empty plate subtracted), and
  all annotations were manually verified.
- `GroundTruth_GTMensa.json` — raw Label Studio export (the file the pipeline
  parses); `structured_food_labels.csv` — same content, one row per item;
  `labels_flat.csv` — convenience version with plain image names (`001.jpg`).
- `GT_Mensa_2025/image_participants.csv` — anonymized image→participant map
  (`participant_1..6`), useful for device/collector-clustered analyses.

All personal data have been removed: images carry no EXIF/GPS metadata,
participants and human raters are identified only by anonymous codes.

## Model outputs

- `model_outputs/` — first campaign: 20 open-source VLM configurations
  (via [Ollama](https://ollama.com)) and OpenAI baselines, 6 temperatures
  × 200 images, raw JSON responses.
- `model_outputs_v2/` — second campaign: 2026-generation models (Qwen3.5,
  Gemma 4, MiniCPM-V 4.x), stochastic replicates, determinism runs, and
  proprietary baselines (GPT-5.5, Claude Sonnet 5). `run_manifest.jsonl`
  records model digests, seeds, timestamps and software versions for every
  run (hostnames anonymized as `machine_N`).

## Code

| Path | Purpose |
|---|---|
| `main.py`, `runpod/main_v2.py` | inference over the image set via Ollama (temperature grid, seeds, JSON output) |
| `openai_baseline.py`, `runpod/openai_v2.py` | proprietary-model baselines |
| `system_prompts/`, `user_prompts/` | the exact prompts used |
| `analysis/evaluation.py` | shared evaluation library: GT loading, robust gram parsing, CrossEncoder scoring, greedy/Hungarian matching |
| `analysis/00…15_*.py` | numbered pipeline: pair scoring, metrics, portion baselines, bootstrap CIs, calibration (LOIO-CV), human-eval form, LLM-as-judge, matcher validation, figures |
| `analysis/outputs/` | all published metrics, bootstrap intervals, judge verdicts, human-rater labels (`rater1..3`), figures |

### Reproducing the analysis

```bash
python -m venv env && source env/bin/activate
pip install pandas numpy scipy matplotlib seaborn pyarrow sentence-transformers
python analysis/01_score_pairs.py      # CrossEncoder pair scores (cached)
python analysis/02_compute_metrics.py  # F1/portion metrics per model/T/eps
python analysis/12_eval_all_v2.py      # same for the v2 campaign
python analysis/04_bootstrap.py        # confidence intervals
```

Re-running inference from scratch requires Ollama ≥ 0.9 with the model tags
listed in the paper (and API keys for the proprietary baselines); all outputs
are already included, so the analysis is fully reproducible offline.

## Licenses

- **Code**: MIT (see `LICENSE`).
- **Dataset** (images, labels, model outputs): Creative Commons Attribution
  4.0 International (see `LICENSE-DATA`).

## Citation

The paper is under review; until it is published, please cite this repository
(see `CITATION.cff`).
