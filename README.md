# AirPulse 🌫️

A self-improving air-quality forecasting system that continuously ingests live AQI and weather data, forecasts AQI 24–48 hours ahead, and **automatically retrains itself** when predictions drift from reality.

> **Portfolio project #6** — demonstrating end-to-end MLOps: data pipelines, CI/CD, containerisation, automated retraining, model versioning/monitoring, and infrastructure-as-code.

---

## What it does

| Capability | Detail |
|---|---|
| **Live ingestion** | Hourly AQI (OpenAQ v3) + weather (Open-Meteo) for 5 Indian cities |
| **Forecasting** | AQI 24–48 h ahead (LightGBM baseline model, $R^2 = 0.93$) |
| **Self-healing** | Statistical drift detection (Evidently AI) triggers automated retraining |
| **Model governance** | Champion/challenger promotion via MLflow Model Registry |
| **Serving** | FastAPI backend + Glassmorphism UI (Chart.js dashboard) |
| **Deployment** | Docker containerization, GHCR registry, k3s manifests & Oracle Cloud VM runbook |

## Target cities

Mumbai · Delhi · Bangalore · Chennai · Kolkata

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Project scaffold & environment | ✅ Done |
| 1 | Data pipeline (ingestion → S3) | ✅ Done |
| 2 | Feature engineering & baseline model | ✅ Done |
| 3 | Serving (FastAPI + frontend + Docker) | ✅ Done |
| 4 | CI/CD (GitHub Actions) | ✅ Done |
| 5 | Drift detection & automated retraining | ✅ Done |
| 6 | Deployment & Orchestration (k3s & Cloud VM) | ✅ Done |

## Repo structure

```
AirPulse/
├── src/
│   ├── ingestion/      # Data ingestion pipelines (OpenAQ & Open-Meteo)
│   ├── features/       # 41 time-series lag/rolling features
│   ├── training/       # LightGBM trainer & MLflow Model Registry logger
│   ├── serving/        # FastAPI backend & Glassmorphism dashboard UI
│   └── monitoring/     # Evidently AI drift detection & retrain trigger
├── infra/
│   ├── terraform/      # AWS S3 & IAM IaC templates
│   ├── docker/         # Dockerfile & Docker Compose
│   └── k8s/            # Kubernetes (k3s) manifests (ConfigMap, Deployment, Service, Ingress, Kustomization)
├── .github/workflows/  # CI/CD pipelines (Pytest & GHCR Docker publish)
├── docs/               # Architecture & Deployment Guide
├── tests/              # 20/20 Passing Pytest unit & integration suite
└── data/               # Local data cache
```

## Setup & Running Locally

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.8 (RTX 5080 / similar)
- Docker & Docker Compose / Kubernetes (`kubectl`)

### Installation & Execution

```bash
# 1. Clone & enter the repo
git clone https://github.com/Aaaaaaaayush/AirPulse.git && cd AirPulse

# 2. Create virtual environment & activate
python -m venv .venv
.venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start local FastAPI server
python -m src.serving.app
```
Access the dashboard at `http://localhost:8000/`.

---

## ☸️ Kubernetes (k3s) & Cloud Deployment

AirPulse includes complete, production-ready Kubernetes manifests in `infra/k8s/`.

### Quick Deployment via Kustomize:

```bash
kubectl apply -k infra/k8s
```

For full step-by-step instructions on deploying to an **Oracle Cloud Always Free VM** with **NGINX**, **Let's Encrypt SSL**, and **DuckDNS**, refer to the [AirPulse Deployment Guide](file:///d:/Projects_Msc/AirPulse/docs/deployment_guide.md).

---

## Hardware

- **Local dev:** NVIDIA RTX 5080 (16 GB VRAM), CUDA 12.8, PyTorch 2.10.0+cu128, Windows
- **Deploy target:** Oracle Cloud Always Free Arm VM (Ubuntu 22.04), k3s / Docker, Nginx, Let's Encrypt SSL, DuckDNS

## License

This project is part of a personal MSc application portfolio.
