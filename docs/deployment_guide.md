# AirPulse Production Deployment & Cloud VM Orchestration Guide

This guide provides step-by-step instructions for deploying **AirPulse** to production using **k3s (Lightweight Kubernetes)** or **Docker Compose** on an **Oracle Cloud Always Free VM** (Ubuntu 22.04 LTS), configured with NGINX Reverse Proxy, free SSL via Let's Encrypt (`certbot`), and free Dynamic DNS via DuckDNS.

---

## 📋 Architecture Overview

```
                      [ Client Web Browsers ]
                                 │
                         (HTTPS : 443 / SSL)
                                 ▼
                     [ DuckDNS Domain / NGINX ]
                                 │
                         (HTTP Proxy : 8000)
                                 ▼
         ┌───────────────────────────────────────────────┐
         │              AirPulse Container               │
         │  FastAPI Serving Engine + Glassmorphism UI   │
         │  LightGBM v4 Model Registry + Evidently AI    │
         └───────────────────────────────────────────────┘
```

---

## 🚀 Option 1: Kubernetes Deployment (`k3s`)

### Prerequisites on Cloud Instance (Ubuntu 22.04 LTS)

1. **Install Lightweight Kubernetes (`k3s`)**:
   ```bash
   curl -sfL https://get.k3s.io | sh -
   sudo chmod 644 /etc/rancher/k3s/k3s.yaml
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
   ```

2. **Verify k3s Node Status**:
   ```bash
   kubectl get nodes
   ```

### Deploying AirPulse via Kustomize

3. **Clone Repository & Apply Manifests**:
   ```bash
   git clone https://github.com/Aaaaaaaayush/AirPulse.git
   cd AirPulse
   kubectl apply -k infra/k8s
   ```

4. **Verify Deployment & Services**:
   ```bash
   kubectl get pods -l app=airpulse-serving
   kubectl get svc airpulse-service
   kubectl get ingress airpulse-ingress
   ```

5. **Test Ingress Connection locally**:
   ```bash
   curl -H "Host: airpulse.local" http://localhost/health
   ```

---

## 🐳 Option 2: Docker Compose Cloud VM Deployment

If deploying directly via Docker Compose on Ubuntu 22.04:

1. **Install Docker & Docker Compose**:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Pull & Launch Image from GitHub Container Registry (GHCR)**:
   ```bash
   docker pull ghcr.io/aaaaaaaayush/airpulse-serving:latest
   docker run -d --name airpulse -p 8000:8000 --restart always ghcr.io/aaaaaaaayush/airpulse-serving:latest
   ```

3. **Verify Container Health**:
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status":"healthy","model_version":"v4","stage":"Production"}
   ```

---

## 🌐 Setting Up NGINX Reverse Proxy + SSL (DuckDNS & Let's Encrypt)

### 1. Set Up DuckDNS (Free Dynamic DNS)

1. Register at [DuckDNS.org](https://www.duckdns.org).
2. Create a domain sub-domain (e.g., `airpulse-demo.duckdns.org`) pointing to your Cloud VM Public IP.

### 2. Configure NGINX Reverse Proxy

1. **Install NGINX**:
   ```bash
   sudo apt install -y nginx certbot python3-certbot-nginx
   ```

2. **Create NGINX Site Configuration** (`/etc/nginx/sites-available/airpulse`):
   ```nginx
   server {
       server_name airpulse-demo.duckdns.org;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **Enable Site & Reload NGINX**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/airpulse /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

### 3. Provision Free SSL Certificate (Certbot)

```bash
sudo certbot --nginx -d airpulse-demo.duckdns.org
```
Certbot will automatically configure HTTPS redirect and set up auto-renewal via systemd timer!

---

## 🔄 Continuous Deployment (CD) Setup

Whenever a push is made to `main`, GitHub Actions automatically builds and pushes the updated image to `ghcr.io/aaaaaaaayush/airpulse-serving:latest`.

To set up auto-pull on your server using Watchtower:
```bash
docker run -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --interval 300 \
  airpulse
```
Watchtower will automatically pull new images and restart the `airpulse` container with zero effort!

---

## 🔍 Verification & Health Checks

- **Main Dashboard**: `https://airpulse-demo.duckdns.org/`
- **Inference API**: `https://airpulse-demo.duckdns.org/predict`
- **Evidently AI Drift Dashboard**: `https://airpulse-demo.duckdns.org/reports/drift_report.html`
- **Health Check Endpoint**: `https://airpulse-demo.duckdns.org/health`
