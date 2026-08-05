# AirPulse 🌫️

A self-improving air-quality forecasting system that continuously ingests live AQI and weather data, forecasts AQI 24–48 hours ahead, and **automatically retrains itself** when predictions drift from reality.

> **Portfolio project #6** — demonstrating end-to-end MLOps: data pipelines, CI/CD, containerisation, automated retraining, model versioning/monitoring, and infrastructure-as-code.

---

## What it does

| Capability | Detail |
|---|---|
| **Live ingestion** | Hourly AQI (OpenAQ v3) + weather (Open-Meteo) for 5 Indian cities |
| **Forecasting** | AQI 24–48 h ahead (LightGBM/XGBoost baseline) |
| **Self-healing** | Drift detection (Evidently AI) triggers automatic retraining |
| **Model governance** | Champion/challenger promotion via MLflow registry |
| **Serving** | FastAPI backend + vanilla HTML/CSS/JS frontend |
| **Deployment** | Docker → k3s on Oracle Cloud VM, CI/CD via GitHub Actions |

## Target cities

Mumbai · Delhi · Bangalore · Chennai · Kolkata

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Project scaffold & environment | ✅ Done |
| 1 | Data pipeline (ingestion → S3) | ✅ Done |
| 2 | Feature engineering & baseline model | ✅ Done |
| 3 | Serving (FastAPI + frontend + Docker) | ⬜ |
| 4 | CI/CD (GitHub Actions) | ⬜ |
| 5 | Drift detection & automated retraining | ⬜ |
| 6 | Orchestration & monitoring (k3s) | ⬜ |
| 7 | Research write-up | ⬜ |

## Repo structure

```
AirPulse/
├── src/
│   ├── ingestion/      # Phase 1 — data pipeline
│   ├── features/       # Phase 2 — feature engineering
│   ├── training/       # Phase 2 — model training
│   ├── serving/        # Phase 3 — FastAPI + frontend
│   │   ├── static/     # HTML/CSS/JS
│   │   └── templates/
│   └── monitoring/     # Phase 5/6 — drift & dashboards
├── infra/
│   ├── terraform/      # Phase 1 — S3, IAM (IaC)
│   └── docker/         # Phase 3 — Dockerfiles
├── .github/workflows/  # Phase 4 — CI/CD
├── docs/               # Documentation & research notes
├── tests/              # Unit & integration tests
├── notebooks/          # EDA notebooks
└── data/               # Local data cache (gitignored)
```

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.8 (RTX 5080 / similar)
- Git

### Installation

```bash
# 1. Clone & enter the repo
git clone <repo-url> && cd AirPulse

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Pre-install PyTorch dependencies (workaround for cu128 metadata bug)
pip install typing-extensions filelock jinja2 sympy fsspec networkx

# 4. Install PyTorch (GPU — must use cu128 wheel index)
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --no-deps

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Copy environment template
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
# Then fill in your API keys and credentials
```

### Verify GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

## Hardware

- **Local dev:** NVIDIA RTX 5080 (16 GB VRAM), CUDA 12.8, PyTorch 2.10.0+cu128, Windows
- **Deploy target:** Oracle Cloud Always Free Arm VM (Ubuntu 22.04), Nginx, systemd, Let's Encrypt SSL, DuckDNS

## License

This project is part of a personal MSc application portfolio.
