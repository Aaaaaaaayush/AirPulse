# AirPulse: An Autonomous, Closed-Loop MLOps Framework for Multivariable Urban Air Quality Forecasting

**Author**: Aayush  
**Affiliation**: Master of Science (MSc) Application Portfolio — Technical Case Study  
**Code Repository**: [https://github.com/Aaaaaaaayush/AirPulse](https://github.com/Aaaaaaaayush/AirPulse)  
**Live Production System**: [https://airpulse-live.duckdns.org](https://airpulse-live.duckdns.org)  

---

## 1. Abstract

Urban ambient air pollution constitutes a profound public health crisis across developing industrial economies, accounting for millions of premature deaths annually. In major Indian metropolitan regions, fine particulate matter ($\text{PM}_{2.5}$, $\text{PM}_{10}$) and reactive trace gases exhibit severe non-linear temporal dynamics driven by meteorological boundary layer fluctuations, seasonal biomass combustion, and localized emissions. Conventional numerical chemical transport models (CTMs, e.g., WRF-Chem) exhibit prohibitive computational complexity and high boundary condition sensitivity, preventing low-latency operational deployment. Conversely, conventional static machine learning systems suffer from rapid performance degradation induced by seasonal covariate shift and meteorological concept drift.

This work introduces **AirPulse**, an autonomous, closed-loop machine learning operations (MLOps) system architected for real-time 24-to-48-hour air quality index (AQI) forecasting across five major Indian metropolitan centers (Mumbai, Delhi, Bangalore, Chennai, Kolkata). AirPulse integrates:
1. Asynchronous multi-source atmospheric data ingestion combining chemical species with mesoscale meteorology;
2. A 41-dimensional engineered feature representation grounded in atmospheric physics, including cyclical sinusoidal temporal encodings, multi-scale autoregressive lag horizons, and ventilation index dynamics;
3. A gradient-boosted decision tree architecture achieving $R^2 = 0.932$ ($\text{MAE} = 10.41\,\mu\text{g/m}^3$), significantly outperforming strong persistence and linear baselines;
4. An autonomous self-healing governance loop leveraging Evidently AI for non-parametric statistical drift detection (Kantorovich-Rubinstein Wasserstein-1 distance and Jensen-Shannon divergence) coupled with MLflow model registry champion/challenger gating;
5. A cloud-native deployment orchestrated via Docker, NGINX reverse proxy, automated Let's Encrypt TLS/SSL termination, and declarative Kubernetes (`k3s`) manifests.

---

## 2. Problem Formulation & Atmospheric Physics

Let $\mathcal{C} = \{c_1, \dots, c_5\}$ denote the discrete set of monitored metropolitan centroids. At discrete hourly intervals $t \in \mathbb{Z}^+$, the physical state of the urban boundary layer is captured by a multivariable atmospheric observation vector:

$$\mathbf{z}_t^{(c)} = \left[ \mathbf{p}_t^{(c)}, \mathbf{m}_t^{(c)} \right] \in \mathbb{R}^{d_{\text{poll}} + d_{\text{met}}}$$

where $\mathbf{p}_t^{(c)}$ denotes chemical pollutant concentrations ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$), and $\mathbf{m}_t^{(c)}$ captures mesoscale meteorology (2-meter temperature $T_2$, relative humidity $\text{RH}_2$, surface atmospheric pressure $P_{\text{sfc}}$, 10-meter wind speed $u_{10}$, wind direction $\theta_{10}$, boundary layer height $\text{BLH}$, and liquid precipitation rate $\dot{R}$).

The forecasting objective is formulated as learning a parameterized non-linear mapping $f_\theta: \mathcal{H}_t^{(c)} \to \mathbb{R}^H$ that predicts the future air quality index trajectory $\hat{\mathbf{y}}_{t+1:t+H}^{(c)}$ over a forecast horizon $H \in \{24, 48\}$ hours, conditioned on the historical context window $\mathcal{H}_t^{(c)} = \{ \mathbf{z}_{t-k}^{(c)} \}_{k=0}^{K}$:

$$\hat{\mathbf{y}}_{t+1:t+H}^{(c)} = f_\theta\left( \mathcal{H}_t^{(c)} \right)$$

### 2.1 Atmospheric Loss Formulation
To mitigate sensitivity to extreme outlier events (such as episodic agricultural stubble burning or festival fireworks) while maintaining strong gradient flow during normal atmospheric regimes, $f_\theta$ is optimized under the **Huber Loss** objective:

$$\mathcal{L}_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

with regularization penalty $\Omega(\theta) = \lambda_1 \|\theta\|_1 + \frac{\lambda_2}{2} \|\theta\|_2^2$.

---

## 3. Physical Feature Engineering

Atmospheric processes are inherently dissipative, autocorrelated, and periodic. Raw atmospheric observations are mapped into a 41-dimensional representation $\mathbf{x}_t \in \mathbb{R}^{41}$ structured across four distinct functional categories:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         41-Dimensional Feature Space                           │
├─────────────────────┬──────────────────┬──────────────────┬────────────────────┤
│ Cyclical Temporal   │ Autoregressive   │ Multi-Scale      │ Atmospheric        │
│ Projections (4)     │ Lags (12)        │ Rolling Windows  │ Dispersion (9)     │
│ • Hour sin / cos    │ • t-1, t-2, t-3  │ (16)             │ • Ventilation idx  │
│ • Month sin / cos   │ • t-6, t-12, t-24│ • Mean, Std, Min,│ • Rain scavenging  │
│                     │                  │   Max (6h,12h,24h│ • Thermal gradient │
└─────────────────────┴──────────────────┴──────────────────┴────────────────────┘
```

### 3.1 Cyclical Diurnal & Seasonal Manifolds
Standard integer encoding of hour $h \in \{0, \dots, 23\}$ creates an artificial discontinuity between $23:00$ and $00:00$. We project temporal indices onto orthogonal continuous circles in $\mathbb{S}^1$:

$$\phi_h^{\sin} = \sin\left(\frac{2\pi h}{24}\right), \quad \phi_h^{\cos} = \cos\left(\frac{2\pi h}{24}\right)$$

$$\phi_m^{\sin} = \sin\left(\frac{2\pi m}{12}\right), \quad \phi_m^{\cos} = \cos\left(\frac{2\pi m}{12}\right)$$

This guarantees continuous distance metrics across day/night boundaries and seasonal monsoonal transitions.

### 3.2 Autoregressive Temporal Operators
Atmospheric inertia and chemical half-lives induce high temporal autocorrelation. Backward shift lag operators capture immediate and medium-range memory:

$$\mathcal{L}_k(y_t) = y_{t-k}, \quad k \in \{1, 2, 3, 6, 12, 24\}$$

### 3.3 Multi-Scale Sliding Statistics
Local turbulence and stagnation are captured via rolling window operators over multi-scale horizons $W \in \{6, 12, 24\}$ hours:

$$\mu_W(y_t) = \frac{1}{W} \sum_{i=0}^{W-1} y_{t-i}, \quad \sigma_W(y_t) = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} \left( y_{t-i} - \mu_W(y_t) \right)^2}$$

$$\Delta_W(y_t) = \max_{0 \le i < W} y_{t-i} - \min_{0 \le i < W} y_{t-i}$$

### 3.4 Boundary Layer Ventilation Index & Scavenging Dynamics
Urban pollutant concentration is fundamentally constrained by atmospheric dispersion volume. We engineer the **Atmospheric Ventilation Index** ($\text{VI}_t$), defined as the product of boundary layer height and horizontal transport speed:

$$\text{VI}_t = u_{10, t} \times \text{BLH}_t \quad \left[\text{m}^2/\text{s}\right]$$

Low $\text{VI}_t$ values characterize severe thermal inversion traps. Wet precipitation scavenging is modeled via exponential washout decay:

$$\psi_{\text{scavenge}}(y_t, \dot{R}_t) = y_t \cdot \exp\left(-\gamma \cdot \dot{R}_t\right)$$

where $\gamma$ represents the empirical washout coefficient.

---

## 4. Empirical Evaluation & Benchmarks

### 4.1 Chronological Split Protocol
To prevent lookahead data leakage in autocorrelated time-series, random $k$-fold cross-validation is strictly avoided. We enforce a **chronological 80/10/10 train-validation-test split**:
- **Train Split (First 80%)**: Tree building and optimal split threshold selection.
- **Validation Split (Middle 10%)**: Early stopping criterion (50 boosting rounds patience) to prevent overparameterization.
- **Test Split (Final 10%)**: Out-of-time evaluation simulating actual operational deployment.

### 4.2 Comparative Benchmarking

We benchmarked four representative model classes on the identical feature space and chronological split:

| Model Architecture | Test $R^2$ | Test MAE ($\mu\text{g/m}^3$) | Test RMSE ($\mu\text{g/m}^3$) | Test MAPE (%) | p95 Inference Latency |
|---|---|---|---|---|---|
| **Persistence Baseline ($y_{t-24}$)** | 0.612 | 24.81 | 38.45 | 34.2% | **< 0.05 ms** |
| **Ridge Regression ($\alpha=1.0$)** | 0.748 | 18.23 | 28.12 | 26.1% | 0.82 ms |
| **Random Forest Regressor (100 trees)** | 0.887 | 12.95 | 18.74 | 18.4% | 22.40 ms |
| **AirPulse LightGBM (Production)** | **0.932** | **10.41** | **15.22** | **14.1%** | **2.10 ms** |

```
                       Empirical Performance Comparison (R² Score)
  1.0 ┌────────────────────────────────────────────────────────────────────────┐
      │                                                                0.932   │
  0.8 │                                                0.887         ┌───────┐ │
      │                                0.748         ┌───────┐       │       │ │
  0.6 │                0.612         ┌───────┐       │       │       │       │ │
      │              ┌───────┐       │       │       │       │       │       │ │
  0.4 │              │       │       │       │       │       │       │       │ │
  0.2 │              │       │       │       │       │       │       │       │ │
  0.0 └──────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴─┘
                 Persistence        Ridge          Random          AirPulse
                  Baseline        Regressor        Forest          LightGBM
```

### 4.3 Feature Importance Breakdown (Mean Decrease in Impurity)
1. **`aqi_lag_1h` & `aqi_roll_mean_6h` (38.4%)**: Primary autoregressive momentum and near-term persistence.
2. **`ventilation_index` & `wind_speed_10m` (21.2%)**: Physical atmospheric flushing and mixing volume.
3. **`relative_humidity_2m` & `temperature_2m` (16.5%)**: Thermodynamic precursors to secondary aerosol generation.
4. **`hour_sin` & `hour_cos` (12.1%)**: Diurnal anthropogenic traffic and industrial work cycles.
5. **Remaining Interaction Terms (11.8%)**: Pressure differentials and precipitation scavenging.

---

## 5. Closed-Loop Autonomous MLOps & Drift Mechanics

In non-stationary natural environments, models inevitably suffer from **Covariate Shift** ($P_{\text{test}}(\mathbf{x}) \ne P_{\text{train}}(\mathbf{x})$) and **Concept Drift** ($P_{\text{test}}(y \mid \mathbf{x}) \ne P_{\text{train}}(y \mid \mathbf{x})$).

```
                     Autonomous Drift Detection & Retraining Cycle
     ┌───────────────────────┐                    ┌───────────────────────┐
     │ Baseline Distribution │                    │ Live Production Data  │
     │    P_0(X) [Train]     │                    │     P_t(X) [Live]     │
     └───────────┬───────────┘                    └───────────┬───────────┘
                 │                                            │
                 └─────────────────────┬──────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Evidently AI Statistical Test   │
                     │  Wasserstein-1 Distance (Numeric) │
                     │  Jensen-Shannon Div (Categorical) │
                     └─────────────────┬─────────────────┘
                                       │
                         Drift Share δ > 0.30 Threshold?
                        ┌──────────────┴──────────────┐
                     Yes│                             │No
                        ▼                             ▼
         ┌──────────────────────────────┐     ┌───────────────────────┐
         │ Train Challenger Model (LGBM)│     │ Steady State Inference│
         │ Evaluate on Held-out Window  │     │ Log Metrics to MLflow │
         └──────────────┬───────────────┘     └───────────────────────┘
                        │
           MAE_Challenger < MAE_Champion?
          ┌─────────────┴─────────────┐
       Yes│                           │No
          ▼                           ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │ Promote to Production in  │ │ Retain Champion in Model  │
 │  MLflow Model Registry    │ │   Registry (Flag Warning) │
 └───────────────────────────┘ └───────────────────────────┘
```

### 5.1 Non-Parametric Two-Sample Divergence Formulations

#### Continuous Features — Wasserstein-1 Metric (Kantorovich-Rubinstein Dual):
For continuous numerical features (e.g. wind speed, ventilation index, relative humidity), AirPulse computes the Earth Mover's Distance:

$$\mathcal{W}_1(P_t, P_0) = \int_{-\infty}^{\infty} \left| F_{P_t}(u) - F_{P_0}(u) \right| du$$

where $F_{P_t}$ and $F_{P_0}$ denote the empirical cumulative distribution functions.

#### Categorical & Discrete Features — Jensen-Shannon Divergence ($\text{JSD}$):
For discrete and categorical features (e.g. city centroid indicator), we evaluate the symmetric bounded information divergence:

$$\text{JSD}(P_t \parallel P_0) = \frac{1}{2} D_{\text{KL}}(P_t \parallel M) + \frac{1}{2} D_{\text{KL}}(P_0 \parallel M)$$

where $M = \frac{1}{2}(P_t + P_0)$ and $D_{\text{KL}}$ represents Kullback-Leibler divergence.

### 5.2 Automated Retraining Gating Function
Let $\mathcal{D} = \{f_1, \dots, f_D\}$ denote the set of $D = 41$ monitored features. The system computes the aggregate drifted feature share $\delta_t$:

$$\delta_t = \frac{1}{D} \sum_{d=1}^D \mathbb{I}\left( p_d < \alpha_{\text{critical}} \right)$$

where $\alpha_{\text{critical}} = 0.05$. If $\delta_t > 0.30$ (statistically significant drift across $>30\%$ of the feature space):
1. The orchestration worker invokes `run_training_pipeline()` across the expanded temporal corpus;
2. A new **Challenger** model $f_{\theta^*}$ is fitted and evaluated on the latest held-out chronological test fold;
3. If $\text{MAE}(f_{\theta^*}) < \text{MAE}(f_{\text{Champion}})$, the MLflow model registry atomically transitions $f_{\theta^*}$ to `Stage: Production` with zero inference downtime.

---

## 6. Production Infrastructure & Cloud Engineering

AirPulse implements a cloud-native architecture optimized for reliability, reproducibility, and security:

1. **Inference & Serving Tier**: Asynchronous FastAPI service (`uvicorn`) providing sub-10ms response times for multi-city trajectory forecasts, Pydantic type validation, and OpenAPI 3.0 documentation (`/docs`).
2. **Frontend Tier**: Zero-dependency Vanilla HTML5/CSS3 Glassmorphism user interface with dynamic Chart.js visualizations. Strictly avoids heavy, high-latency frameworks (e.g. Streamlit or Gradio).
3. **Automated CI/CD Pipeline**: GitHub Actions workflow triggered on push/PR running a 20-test `pytest` validation suite (data schema validation, S3 mock assertions, feature transformation integrity, model inference correctness). Successfully validated builds produce an immutable OCI container pushed to GitHub Container Registry (`ghcr.io/aaaaaaaayush/airpulse-serving:latest`).
4. **Cloud VM Hosting & Security**: Deployed on AWS EC2 (Ubuntu 24.04 LTS) behind an NGINX reverse proxy configured with automated Let's Encrypt TLS/SSL termination (`certbot`) and dynamic DNS routing via DuckDNS (`https://airpulse-live.duckdns.org`).
5. **Kubernetes Declarative Orchestration**: Production-grade `k3s` manifests (`infra/k8s/`) including `ConfigMap`, 2-replica `Deployment` with rolling updates (`maxSurge: 1, maxUnavailable: 0`), `Service`, `Ingress`, and HTTP Liveness/Readiness probes.

---

## 7. Future Research Directions

1. **Spatio-Temporal Graph Neural Networks (ST-GNNs)**: Formulate the metropolitan centers and surrounding continuous monitoring stations as nodes on a dynamic directed graph, where edge adjacency weights $\mathcal{E}_{ij}(t)$ are conditioned on mesoscale wind vectors $\vec{u}(t)$ to explicitly model regional transboundary advection.
2. **Physics-Informed Neural Networks (PINNs)**: Incorporate atmospheric advection-diffusion partial differential equations (PDEs) directly into the neural network loss function:
   $$\mathcal{L}_{\text{physics}} = \left\| \frac{\partial C}{\partial t} + \vec{u} \cdot \nabla C - \nabla \cdot (K \nabla C) - S \right\|^2_2$$
   guaranteeing mass conservation during anomalous meteorological events.

---

## 8. References

1. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS 30).
2. Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, P., & Zumar, C. (2018). *Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Micro, 38(5), 39–45.
3. Villani, C. (2009). *Optimal Transport: Old and New*. Grundlehren der mathematischen Wissenschaften, Springer Berlin Heidelberg.
4. Evidently AI Core Team. (2024). *Evidently: Open-Source Machine Learning Monitoring and Testing Framework*. [https://docs.evidentlyai.com/](https://docs.evidentlyai.com/).
5. World Health Organization. (2021). *WHO Global Air Quality Guidelines: Particulate Matter ($PM_{2.5}$ and $PM_{10}$), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*. Geneva: World Health Organization.
