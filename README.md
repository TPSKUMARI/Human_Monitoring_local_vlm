# Human Monitoring - Local VLM

A local, real-time person monitoring system that uses a Vision-Language Model (LLaVA) to watch a webcam feed and generate safety/wellness assessments — no cloud API required.

## What it does

- Captures webcam frames continuously and feeds them to a local VLM (`llava-hf/llava-interleave-qwen-0.5b-hf`) every few seconds.
- Produces a 7-point analysis covering person detection, activity, safety risks, health/wellness indicators, environmental hazards, behavioral state, and care recommendations.
- Classifies each analysis into an activity type (e.g. sitting, walking, lying down) and a safety level (safe / normal / medium_risk / high_risk).
- Logs every analysis to a local SQLite database (`person_monitoring.db`).
- Automatically records 1-minute video clips every 4 minutes to the `output/` folder, with a live info overlay.
- Displays a live window with timestamp, activity, safety level, and recording status.

## Requirements

- Python 3.9+
- A webcam
- (Recommended) NVIDIA GPU with CUDA for faster inference — falls back to CPU otherwise

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python Local_vlm_human_monitoring.py
```

Controls while the monitoring window is active:

- `q` — quit
- `s` — print the latest detailed analysis
- `r` — print recent activities from the database (last 24 hours)
- `v` — manually start a video recording

## Output

- **Database**: `person_monitoring.db` (recreated fresh on each run)
- **Videos**: saved under `output/monitoring_<timestamp>.mp4`

## Notes

This is a simple, single-file prototype intended for local experimentation, not production deployment.
