# AirPulse 🌫️

[![Live Demo](https://img.shields.io/badge/Live%20Demo-airpulse--live.duckdns.org-brightgreen?style=for-the-badge&logo=nginx)](https://airpulse-live.duckdns.org)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20Passing-success?style=for-the-badge&logo=githubactions)](https://github.com/Aaaaaaaayush/AirPulse/actions)
[![Model](https://img.shields.io/badge/Model-LightGBM%20v4%20(R%C2%B2%200.932)-blue?style=for-the-badge&logo=scikitlearn)](docs/research_writeup.md)
[![Registry](https://img.shields.io/badge/Container-GHCR%20Docker-blue?style=for-the-badge&logo=docker)](https://github.com/Aaaaaaaayush/AirPulse/pkgs/container/airpulse-serving)
[![Monitoring](https://img.shields.io/badge/Drift%20Monitoring-Evidently%20AI-orange?style=for-the-badge)](https://airpulse-live.duckdns.org/reports/drift_report.html)

A self-improving, production-grade air-quality forecasting system that continuously ingests real-time meteorological and AQI data, forecasts air pollution 24–48 hours ahead, and **autonomously retrains itself** when real-world distributions drift.

> **Project intention** — demonstrating an end-to-end MLOps lifecycle: multi-source asynchronous ingestion pipelines, 41-feature time-series engineering, LightGBM model governance via MLflow, statistical drift detection with Evidently AI, automated GitHub Actions CI/CD to GHCR, and live cloud deployment on AWS EC2 with NGINX, Let's Encrypt TLS/SSL, and DuckDNS.

---

## 🌐 Live Production Links

- **Live Web Dashboard**: [https://airpulse-live.duckdns.org/](https://airpulse-live.duckdns.org/)
- **Interactive Evidently AI Drift Report**: [https://airpulse-live.duckdns.org/reports/drift_report.html](https://airpulse-live.duckdns.org/reports/drift_report.html)
- **FastAPI OpenAPI Interactive Docs**: [https://airpulse-live.duckdns.org/docs](https://airpulse-live.duckdns.org/docs)
- **Container Health Check**: [https://airpulse-live.duckdns.org/health](https://airpulse-live.duckdns.org/health)
- **In-Depth Research & Technical Case Study**: [`docs/research_writeup.md`](docs/research_writeup.md)
- **Production Cloud Runbook**: [`docs/deployment_guide.md`](docs/deployment_guide.md)

---

## 🏙️ Target Metropolitan Cities

**Mumbai** · **Delhi** · **Bangalore** · **Chennai** · **Kolkata**

---

## ⚡ Architecture & Closed-Loop MLOps

```
                     ┌─────────────────────────────────────────┐
                     │          Live Ingestion Tier            │
                     │  • Open-Meteo Weather & AQI APIs        │
                     │  • OpenAQ v3 Atmospheric Sensors        │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │       Feature Engineering Engine        │
                     │  • 41 Lag, Rolling & Diurnal Features   │
                     │  • Strict Chronological Validation      │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │       Model Training & Governance       │
                     │  • LightGBM Regressor (R² = 0.932)      │
                     │  • MLflow Champion/Challenger Registry  │
                     └─────────┬─────────────────────▲─────────┘
                               │                     │
               Model Artifacts │                     │ Auto-Retrain
                               ▼                     │ Trigger
                     ┌──────────────────┐    ┌───────┴─────────┐
                     │ FastAPI Serving  │    │   Evidently AI  │
                     │ Docker Container │───>│ Drift Detection │
                     └─────────┬────────┘    │  (Wasserstein / │
                               │             │ Jensen-Shannon) │
                               ▼             └─────────────────┘
                     ┌──────────────────┐
                     │ NGINX / Cloud VM │
                     │ HTTPS / DuckDNS  │
                     └──────────────────┘
```

---

## 📊 Project Phases & Roadmap

| Phase | Focus | Status | Deliverables |
|---|---|---|---|
| **0** | Project scaffold & environment | ✅ Done | Directory layout, PyTorch GPU cu128 verification, `.env.example` |
| **1** | Data pipeline (ingestion → S3) | ✅ Done | Async Open-Meteo & OpenAQ clients, S3 storage, Terraform IaC |
| **2** | Feature engineering & baseline model | ✅ Done | 41 time-series features, LightGBM ($R^2 = 0.932$), MLflow Model Registry |
| **3** | Serving (FastAPI + Glassmorphism UI) | ✅ Done | FastAPI async server, Vanilla CSS Glassmorphism dashboard, Chart.js |
| **4** | CI/CD (GitHub Actions) | ✅ Done | Automated 20/20 test suite on commit, automated GHCR Docker container build |
| **5** | Drift detection & automated retraining | ✅ Done | Evidently AI statistical drift tests, `/api/drift`, HTML report serving |
| **6** | Deployment & Cloud Orchestration | ✅ Done | AWS EC2 Ubuntu instance, NGINX reverse proxy, Let's Encrypt SSL, k3s manifests |
| **7** | Research write-up & Portfolio Showcase | ✅ Done | Academic case study (`docs/research_writeup.md`), badges, release tag |

---

## 📈 Model Performance & Benchmarks

Strict chronological 80/10/10 evaluation preventing forward lookahead leakage:

| Architecture | $R^2$ Score | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | Inference Latency |
|---|---|---|---|---|
| **Persistence Baseline ($t-24$)** | 0.612 | 24.81 | 38.45 | < 0.1 ms |
| **Ridge Linear Model** | 0.748 | 18.23 | 28.12 | 0.8 ms |
| **Random Forest (100 trees)** | 0.887 | 12.95 | 18.74 | 22.4 ms |
| **AirPulse LightGBM (Production)** | **0.932** | **10.41** | **15.22** | **2.1 ms** |

---

## 🛠️ Repository Structure

```
AirPulse/
├── src/
│   ├── ingestion/      # Multi-source asynchronous data ingestion pipelines
│   ├── features/       # 41-feature time-series transformer (lags, rolling stats, diurnal)
│   ├── training/       # LightGBM trainer & MLflow Model Registry integration
│   ├── serving/        # FastAPI application & modern Glassmorphism dashboard UI
│   └── monitoring/     # Evidently AI statistical drift detector & retraining triggers
├── infra/
│   ├── terraform/      # AWS S3 & IAM Infrastructure as Code
│   ├── docker/         # Production Dockerfile & Docker Compose
│   └── k8s/            # Kubernetes manifests (ConfigMap, Deployment, Service, Ingress)
├── .github/workflows/  # CI/CD pipelines (Pytest runner & GHCR container publisher)
├── docs/               # Research write-up and production deployment guide
├── tests/              # 20/20 Passing Pytest unit & integration test suite
└── data/               # Local data artifacts cache (gitignored)
```

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.10+
- Git
- Docker / Docker Compose

### Quickstart
```bash
# 1. Clone repository
git clone https://github.com/Aaaaaaaayush/AirPulse.git
cd AirPulse

# 2. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/ -v

# 5. Launch local serving app
python -m src.serving.app
```
Visit `http://localhost:8000/` for the dashboard and `http://localhost:8000/docs` for API specifications.

---

## ☸️ Cloud VM & Kubernetes (`k3s`) Deployment

To deploy directly to a Kubernetes cluster via Kustomize:
```bash
kubectl apply -k infra/k8s
```

Or deploy via Docker container:
```bash
docker run -d --name airpulse -p 8000:8000 --restart always ghcr.io/aaaaaaaayush/airpulse-serving:latest
```

Full cloud setup (AWS EC2, NGINX reverse proxy, Let's Encrypt SSL, DuckDNS) is documented in [docs/deployment_guide.md](docs/deployment_guide.md).

---

## 📜 License & Citation

This project is open-source under the MIT License and was developed as part of an MSc application portfolio in Advanced Computer Science / Machine Learning Systems.
