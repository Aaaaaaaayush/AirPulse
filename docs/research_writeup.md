# AirPulse: An Autonomous, Closed-Loop MLOps Framework for Multivariable Urban Air Quality Forecasting

**Author**: Aayush  
**Repository**: [https://github.com/Aaaaaaaayush/AirPulse](https://github.com/Aaaaaaaayush/AirPulse)  
**Live Production Deployment**: [https://airpulse-live.duckdns.org](https://airpulse-live.duckdns.org)  
**Target Focus**: MSc Application Portfolio Project #6 (Machine Learning Systems & Applied AI)

---

## 1. Abstract

Urban air pollution represents one of the most critical public health hazards across rapidly developing economies. In Indian metropolitan agglomerations, particulate matter ($\text{PM}_{2.5}, \text{PM}_{10}$) and associated pollutants exhibit severe non-linear temporal dynamics driven by meteorological boundary layer fluctuations, seasonal biomass combustion, and vehicular emissions. Traditional numerical air quality models (e.g., WRF-Chem) suffer from prohibitive computational latency and boundary condition sensitivity, while naive machine learning models degrade rapidly due to seasonal covariate shift and concept drift. 

This paper presents **AirPulse**, an end-to-end autonomous, self-improving machine learning operations (MLOps) system designed for real-time, 24-to-48-hour air quality index (AQI) forecasting across five major Indian metropolitan centers (Mumbai, Delhi, Bangalore, Chennai, Kolkata). AirPulse integrates asynchronous multi-source ingestion, a 41-dimensional engineered feature space incorporating cyclical diurnal encodings and meteorological interactions, a gradient-boosted decision tree architecture ($R^2 = 0.932$, $\text{MAE} = 10.41$), and an automated closed-loop statistical drift detection engine powered by Evidently AI and MLflow. The entire system is containerized and deployed in production on cloud infrastructure with automated CI/CD and Let's Encrypt TLS encryption.

---

## 2. System Architecture

AirPulse is designed around the principle of **autonomous self-healing**: a continuous feedback loop that monitors model performance against ground truth and adapts to seasonal climate transitions without human intervention.

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
                     │  • Chronological Cross-Validation       │
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

## 3. Data Ingestion & Atmospheric Variables

Air quality forecasting requires combining atmospheric chemistry with mesoscale meteorological dynamics. AirPulse ingests hourly observations spanning:

1. **Particulate & Gaseous Pollutants**: $\text{PM}_{2.5}$, $\text{PM}_{10}$, Nitrogen Dioxide ($\text{NO}_2$), Sulfur Dioxide ($\text{SO}_2$), Carbon Monoxide ($\text{CO}$), and Ozone ($\text{O}_3$).
2. **Boundary Layer Meteorology**: 2-meter surface temperature ($T$), relative humidity ($\text{RH}$), surface pressure ($P$), wind speed ($u$), wind direction ($\theta$), boundary layer height ($\text{BLH}$), and precipitation.

Data is fetched via asynchronous non-blocking HTTP clients (`httpx`, `asyncio`) with exponential backoff and localized disk/S3 caching (`s3://airpulse-raw/`) for historical model retraining.

---

## 4. Mathematical Feature Engineering

Raw atmospheric variables cannot directly capture temporal hysteresis, diurnal cycles, or ventilation dynamics. We construct a 41-dimensional feature vector $\mathbf{x}_t$ for each timestamp $t$:

### 4.1 Cyclical Diurnal & Seasonal Encodings
To preserve the cyclical topology of diurnal and annual cycles without artificial boundary discontinuities, hour-of-day $h \in [0, 23]$ and month $m \in [1, 12]$ are projected onto orthogonal sinusoidal coordinates:

$$\phi_{\text{hour}}^{\sin} = \sin\left(\frac{2\pi h}{24}\right), \quad \phi_{\text{hour}}^{\cos} = \cos\left(\frac{2\pi h}{24}\right)$$

$$\phi_{\text{month}}^{\sin} = \sin\left(\frac{2\pi m}{12}\right), \quad \phi_{\text{month}}^{\cos} = \cos\left(\frac{2\pi m}{12}\right)$$

### 4.2 Autoregressive Lag Embeddings
Pollutant accumulation exhibits strong autocorrelation. We construct lag operators across short and medium horizons:

$$\mathcal{L}_k(y_t) = y_{t-k}, \quad k \in \{1, 2, 3, 6, 12, 24\}$$

### 4.3 Multi-Scale Rolling Statistics
Local atmospheric stability and dispersion are modeled using sliding window aggregations over window sizes $W \in \{6, 12, 24\}$ hours:

$$\mu_W(y_t) = \frac{1}{W} \sum_{i=0}^{W-1} y_{t-i}, \quad \sigma_W(y_t) = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} (y_{t-i} - \mu_W(y_t))^2}$$

$$\text{Range}_W(y_t) = \max_{i \in [0, W-1]} y_{t-i} - \min_{i \in [0, W-1]} y_{t-i}$$

### 4.4 Atmospheric Dispersion Interactions
Urban air stagnation is governed by the atmospheric **Ventilation Index** ($\text{VI}$), defined as the product of wind speed and mixing height:

$$\text{VI}_t = u_t \times \text{BLH}_t$$

Additionally, precipitation scrubbing is captured via exponential wet deposition decay interaction terms:

$$\kappa_{\text{wet}} = y_t \cdot \exp\left(-\lambda_{\text{rain}} \cdot \text{Precip}_t\right)$$

---

## 5. Model Architecture & Empirical Evaluation

### 5.1 Chronological Split & Leakage Prevention
Because air quality series exhibit severe temporal autocorrelation, standard random train/test splitting introduces severe optimistic data leakage. AirPulse enforces a strict **chronological 80/10/10 split**:
- **Train Split (80%)**: Used for tree induction and split point selection.
- **Validation Split (10%)**: Used for early stopping (patience = 50 rounds).
- **Test Split (10%)**: Out-of-time evaluation simulating real-world forward prediction.

### 5.2 Model Benchmarks & Comparison

We benchmarked multiple predictive architectures on identical feature representations:

| Architecture | $R^2$ Score | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | Inference Latency (p95) |
|---|---|---|---|---|
| **Persistence Baseline ($t-24$)** | 0.612 | 24.81 | 38.45 | < 0.1 ms |
| **Ridge Regularized Linear Model** | 0.748 | 18.23 | 28.12 | 0.8 ms |
| **Random Forest Regressor (100 trees)** | 0.887 | 12.95 | 18.74 | 22.4 ms |
| **AirPulse LightGBM (Selected)** | **0.932** | **10.41** | **15.22** | **2.1 ms** |

### 5.3 Feature Importance Analysis
Gini gain analysis across LightGBM splits reveals the primary predictive drivers:
1. `aqi_lag_1h` and `aqi_roll_mean_6h` (Primary autoregressive inertia — 38.4% importance).
2. `ventilation_index` and `wind_speed_10m` (Atmospheric dispersion capacity — 21.2% importance).
3. `relative_humidity_2m` and `temperature_2m` (Photochemical secondary aerosol formation — 16.5% importance).
4. `hour_sin` and `hour_cos` (Diurnal traffic rush-hour emissions — 12.1% importance).

---

## 6. Autonomous Self-Improving Loop (Evidently AI)

In operational settings, atmospheric distributions shift due to seasonal monsoons, agricultural crop burning (e.g. northern stubble burning), and urban emission shifts.

### 6.1 Statistical Drift Metrics
AirPulse monitors data drift using non-parametric two-sample divergence tests:
- **Continuous Features**: Two-sample **Wasserstein-1 Distance** (Earth Mover's Distance):
  $$W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| dx$$
- **Categorical / Discrete Features**: **Jensen-Shannon Divergence** ($\text{JSD}$):
  $$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$

### 6.2 Closed-Loop Retraining Decision Rule
The drift detection module executes periodically or via automated daily cron:

$$\delta = \frac{1}{D} \sum_{d=1}^D \mathbb{I}\left(p_d < \alpha_{\text{drift}}\right)$$

Where $D = 41$ features, $\alpha_{\text{drift}} = 0.05$. If the drifted feature share $\delta > 0.30$ (30% threshold):
1. The orchestrator triggers `run_training_pipeline()` on the newly accumulated historical corpus.
2. A new **Challenger model** is trained and evaluated on held-out validation data.
3. If $\text{MAE}_{\text{Challenger}} < \text{MAE}_{\text{Champion}}$, MLflow automatically transitions the model state to `Stage: Production` with zero downtime.

---

## 7. Cloud Deployment & Production Engineering

The production stack emphasizes high throughput, low latency, and operational reproducibility:

- **Serving Tier**: FastAPI application utilizing asynchronous endpoint routing and Pydantic schema validation.
- **Frontend Tier**: Custom Vanilla HTML5 / Modern CSS3 Glassmorphism dashboard rendered with Chart.js. Zero reliance on heavy abstractions like Streamlit or Gradio.
- **Continuous Integration / Continuous Deployment (CI/CD)**: GitHub Actions workflow executing 20/20 unit/integration tests with `pytest` on every commit and building an OCI-compliant container image pushed to GitHub Container Registry (`ghcr.io/aaaaaaaayush/airpulse-serving:latest`).
- **Cloud Infrastructure**: Hosted on an AWS EC2 instance behind an NGINX reverse proxy with automated Let's Encrypt TLS/SSL termination (`certbot`) and dynamic DNS resolution via DuckDNS (`https://airpulse-live.duckdns.org`).
- **Kubernetes Orchestration**: Declarative `k3s` manifests (`ConfigMap`, `Deployment`, `Service`, `Ingress`, `Kustomization`) providing 2-replica high availability with automated Liveness and Readiness health probes.

---

## 8. Conclusion & Future Research Directions

AirPulse demonstrates that combining domain-specific meteorological feature engineering with modern MLOps governance principles yields an air quality forecasting system that is both highly accurate ($R^2 = 0.932$) and resilient to temporal distribution shift. 

Promising avenues for future extension include:
1. **Spatio-Temporal Graph Neural Networks (ST-GNNs)**: Formulating the five target cities and intermediate atmospheric stations as graph nodes connected by wind vector adjacency matrices to model trans-boundary pollutant advection.
2. **Physics-Informed Neural Networks (PINNs)**: Imposing atmospheric diffusion partial differential equation (PDE) constraints directly into the loss function to guarantee physical consistency during anomalous extreme weather events.

---

## 9. References

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
3. World Health Organization (2021). *WHO Global Air Quality Guidelines: Particulate Matter and Ozone*.
4. Evidently AI Documentation (2024). *Data Drift and Model Monitoring Framework*.
5. Zaharia, M., et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Micro.
