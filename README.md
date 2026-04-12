# Stress Detection System

A production-grade, end-to-end MLOps system for real-time physiological stress
detection. Multiple simulated patients stream EDA + BVP signals to a cloud-deployed
CNN+LSTM model. Predictions are enriched by an LLM (Groq) and routed through a
doctor-in-the-loop approval workflow before reaching the patient.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          SENSOR LAYER                                │
│  simulator(P001) ──┐                                                 │
│  simulator(P002) ──┼──► POST /predict  {patient_id, EDA[], BVP[]}   │
│  simulator(P003) ──┘                                                 │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE SERVICE  (:8000)                     │
│  1. Preprocess signal window                                        │
│  2. CNN+LSTM → {label, confidence}                                  │
│  3. Call LLM service with {label, confidence}  ← no patient data    │
│  4. Assemble full payload                                           │
│  5. Forward to response-router                                      │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ calls
                   ▼
┌──────────────────────────────┐
│    LLM SERVICE  (:8001)      │
│  In:  {label, confidence}    │
│  Out: {patient_msg,          │
│        doctor_msg}           │
│  Via: Groq API (LLaMA 3)     │
└──────────────┬───────────────┘
               │ response returns to inference service
               │
┌──────────────▼──────────────────────────────────────────────────────┐
│                    RESPONSE ROUTER  (:8002)                         │
│                                                                     │
│  PATH A — immediate (no doctor needed)                              │
│    generic_msg = static lookup by label                             │
│    → dispatch to patient WS instantly                               │
│    → log phase1_sent_at to PostgreSQL                               │
│                                                                     │
│  PATH B — deferred (doctor-gated)                                   │
│    → write full record to PostgreSQL  (status: PENDING)             │
│    → notify doctor WS with full payload + generic_msg_already_sent  │
│    → wait for doctor: APPROVE / MODIFY / REJECT                     │
│    → on approval → dispatch final_patient_msg to patient WS         │
│    → update DB: phase2_sent_at, final_patient_msg, approval details │
└──────────┬──────────────────────────────────────┬───────────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐              ┌───────────────────────────────┐
│  PATIENT  WS         │              │  DOCTOR DASHBOARD  WS         │
│                      │              │                               │
│  Phase 1 (instant):  │              │  Per patient:                 │
│  "Take a deep        │              │  label, confidence,           │
│   breath..."         │              │  doctor_msg, patient_msg,     │
│                      │              │  generic_msg already sent     │
│  Phase 2 (deferred): │              │                               │
│  Personalised msg    │              │  [APPROVE] [MODIFY] [REJECT]  │
│  after doctor OK     │              │                               │
└──────────────────────┘              └───────────────────────────────┘
```

---

## Services

| Service | Port | Description |
|---|---|---|
| `inference-service` | 8000 | FastAPI — CNN+LSTM inference, model loading from MLflow |
| `llm-service` | 8001 | FastAPI — Groq API wrapper, purely generative, stateless |
| `response-router` | 8002 | FastAPI — Two-phase dispatch, WebSocket hub, doctor review API |
| `sensor-simulator` | — | Async Python — N concurrent patient signal streams |
| `postgres` | 5432 | PostgreSQL 16 — Inference log, approval records |
| `minio` | 9000/9001 | MinIO — S3-compatible MLflow artifact store |
| `mlflow` | 5000 | MLflow server — Experiment tracking, model registry |
| `prometheus` | 9090 | Prometheus — Metrics scraping |
| `grafana` | 3000 | Grafana — Dashboards and alerting |

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML Framework | PyTorch |
| Model Architecture | CNN + LSTM (binary stress classification) |
| Dataset | WESAD (EDA + BVP, wrist sensor) |
| API Framework | FastAPI + Uvicorn |
| LLM | Groq API (LLaMA 3 — llama3-8b-8192) |
| Experiment Tracking | MLflow |
| Artifact Store | MinIO (S3-compatible) |
| Database | PostgreSQL 16 + SQLAlchemy (async) + Alembic |
| Containerisation | Docker + Docker Compose |
| Orchestration | Kubernetes (Minikube on M1) |
| Autoscaling | HPA — inference-service (1→5 pods), llm-service (1→3 pods) |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions + GHCR |
| Linting | Ruff |
| Testing | Pytest + pytest-asyncio |

---

## Project Structure

```
stress-detection-system/
├── services/
│   ├── inference-service/       # FastAPI + PyTorch model loading
│   │   ├── app/
│   │   │   ├── main.py          # Routes: /predict, /health, /metrics, /model/info
│   │   │   ├── model.py         # CNN+LSTM model definition + MLflow loader
│   │   │   ├── preprocess.py    # Signal windowing + normalisation
│   │   │   └── metrics.py       # Prometheus instrumentation
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── llm-service/             # Groq API wrapper
│   │   ├── app/
│   │   │   ├── main.py          # Route: /generate
│   │   │   └── prompt.py        # Prompt templates
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── response-router/         # Two-phase dispatch + doctor review
│   │   ├── app/
│   │   │   ├── main.py          # Routes + WebSocket endpoints
│   │   │   ├── router.py        # Phase 1 + Phase 2 dispatch logic
│   │   │   ├── connections.py   # WebSocket connection registry
│   │   │   ├── review.py        # Doctor approval endpoints
│   │   │   └── models.py        # SQLAlchemy ORM models
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── sensor-simulator/        # Async multi-patient signal generator
│   │   ├── simulator.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── retraining-pipeline/     # Training + MLflow logging
│       ├── train.py
│       ├── evaluate.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── ml/
│   ├── data/
│   │   ├── wesad_loader.py      # WESAD .pkl parser
│   │   └── preprocess.py        # Windowing, normalisation, train/val split
│   ├── models/
│   │   └── cnn_lstm.py          # CNN+LSTM architecture definition
│   └── experiments/
│       └── train_baseline.py    # Training entrypoint
│
├── k8s/                         # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmaps.yaml
│   ├── secrets.template.yaml    # Template only — real values via CI/CD
│   ├── ingress.yaml
│   ├── inference-service/deployment.yaml   # Deployment + Service + HPA
│   ├── llm-service/deployment.yaml
│   ├── response-router/deployment.yaml
│   ├── postgres/deployment.yaml
│   ├── minio/deployment.yaml
│   └── monitoring/
│       ├── prometheus/
│       └── grafana/
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
│
├── db/
│   ├── init.sql                 # Initial schema + patient seed data
│   └── migrations/              # Alembic migrations (Phase 5c)
│
├── .github/workflows/
│   ├── ci-cd.yml                # Lint → Test → Build → Push → Deploy
│   └── retrain.yml              # Scheduled / manual retraining trigger
│
├── docker-compose.yml           # Full local stack
├── setup.sh                     # Phase 1 environment setup (M1 Mac)
├── stress-detection.code-workspace
├── .env.example
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- macOS M1 (Apple Silicon)
- Python 3.11+
- Docker Desktop for Apple Silicon
- Homebrew

### 1. Clone and setup

```bash
git clone https://github.com/YOUR_USERNAME/stress-detection-system.git
cd stress-detection-system
chmod +x setup.sh && ./setup.sh
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — minimum required:
#   GROQ_API_KEY=your_key_from_console.groq.com
```

### 3. Download WESAD dataset

Request access at:
https://uni-siegen.de/life-science-informatics/ressourcen/software/wesad/

Extract to `ml/data/wesad/` — expected structure:
```
ml/data/wesad/
  S2/S2.pkl
  S3/S3.pkl
  S4/S4.pkl
  ...  (S2–S17, S12 excluded — corrupted)
```

### 4. Open workspace in VSCode

```bash
code stress-detection.code-workspace
```

### 5. Run local stack (after Phase 3+)

```bash
docker compose up --build
```

---

## Data Flow — Detailed

### Signal characteristics (WESAD wrist sensor)

| Signal | Sampling Rate | Window (60s) | Samples per window |
|---|---|---|---|
| EDA (Electrodermal Activity) | 4 Hz | 60s | 240 |
| BVP (Blood Volume Pulse) | 64 Hz | 60s | 3840 |

### Labels (binary)

| Label | Meaning | WESAD original label(s) |
|---|---|---|
| `STRESS` | Stress detected | 2 (stress) |
| `NO_STRESS` | No stress | 1 (baseline), 3 (amusement) |

### Message routing

| Event | Who receives | When |
|---|---|---|
| Phase 1 generic message | Patient | Immediately (~1s after inference) |
| Doctor notification | Doctor | Immediately after Phase 1 |
| Phase 2 personalised message | Patient | After doctor approves/modifies |

---

## Database Schema

```sql
inference_log
─────────────────────────────────────────────────────
id                  UUID          PK
patient_id          VARCHAR(64)   indexed
timestamp           TIMESTAMPTZ   indexed
eda_window          JSONB
bvp_window          JSONB
label               VARCHAR(16)   STRESS | NO_STRESS
confidence          FLOAT
model_version       VARCHAR(64)
patient_msg         TEXT          LLM generated
doctor_msg          TEXT          LLM generated
generic_msg         TEXT          static lookup
phase1_sent_at      TIMESTAMPTZ
approval_status     VARCHAR(16)   PENDING|APPROVED|MODIFIED|REJECTED
approved_by         VARCHAR(64)
approved_at         TIMESTAMPTZ
modified_msg        TEXT
final_patient_msg   TEXT
phase2_sent_at      TIMESTAMPTZ
reject_reason       TEXT
```

---

## Monitoring Metrics

| Metric | Service | Meaning |
|---|---|---|
| `http_requests_total` | All | Request count by endpoint |
| `http_request_duration_seconds` | All | Latency histogram |
| `inference_prediction_total` | inference | Prediction count by label |
| `inference_model_version` | inference | Currently loaded model version |
| `llm_request_duration_seconds` | llm | Groq API call latency |
| `router_pending_queue_depth` | router | Records awaiting doctor review |
| `router_phase1_dispatch_ms` | router | Time to send generic message |
| `router_phase2_dispatch_seconds` | router | Time from PENDING to approval |
| `router_approval_total` | router | Count by status (approved/modified/rejected) |

---

## CI/CD Pipeline

```
Push to main
    │
    ├─ 1. Lint (ruff) — all services in parallel
    ├─ 2. Unit tests (pytest) — per service
    ├─ 3. Build Docker images (linux/amd64 + linux/arm64)
    ├─ 4. Push to GHCR
    ├─ 5. kubectl apply k8s manifests
    └─ 6. Smoke test /health endpoints
```

Retraining workflow runs on schedule (weekly) or manual trigger.

---

## Kubernetes — Minikube Setup (M1)

```bash
# Start cluster (6GB RAM — leaves 2GB for macOS)
minikube start --driver=docker --cpus=4 --memory=6144

# Enable ingress
minikube addons enable ingress

# Enable metrics-server (required for HPA)
minikube addons enable metrics-server

# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps.yaml
kubectl apply -f k8s/secrets.template.yaml  # fill values first
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/minio/
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/inference-service/
kubectl apply -f k8s/llm-service/
kubectl apply -f k8s/response-router/
kubectl apply -f k8s/sensor-simulator/
kubectl apply -f k8s/monitoring/
kubectl apply -f k8s/ingress.yaml

# Watch HPA
kubectl get hpa -n stress-system --watch
```

---

## Phase Execution Plan

| Phase | Focus | Status |
|---|---|---|
| 1 | Repo setup, tooling, environment config | ✅ Complete |
| 2 | WESAD loader, preprocessing, CNN+LSTM training, MLflow | ⬜ Next |
| 3 | Inference service + Dockerize | ⬜ |
| 4 | Sensor simulator (multi-patient concurrent) | ⬜ |
| 5a | LLM service + Groq | ⬜ |
| 5b | Response router + WebSocket dispatch | ⬜ |
| 5c | PostgreSQL + SQLAlchemy + Alembic | ⬜ |
| 6 | Docker Compose full local stack | ⬜ |
| 7 | MinIO migration for MLflow artifacts | ⬜ |
| 8 | Prometheus + Grafana dashboards | ⬜ |
| 9 | Kubernetes + HPA on Minikube | ⬜ |
| 10 | GitHub Actions CI/CD | ⬜ |
| 11 | Retraining pipeline | ⬜ |
| 12 | Integration test + documentation | ⬜ |

---

## Security Notes

- Raw EDA/BVP signal windows never leave the inference service
- Groq API receives only `{label, confidence}` — no patient identifiers
- Patient messages never reach patients without doctor approval (Phase 2)
- All secrets managed via Kubernetes Secrets / GitHub Actions secrets
- Non-root user inside all Docker containers
- `.env` is gitignored — only `.env.example` is committed
- In a real clinical deployment, a HIPAA BAA with the cloud provider
  would be required before processing real patient data

---

## Future — v2 (Edge Inference Track)

- Edge device runs the CNN+LSTM model locally (no raw signals leave device)
- Only the prediction result `{label, confidence}` is sent to the cloud
- Model updates delivered via authenticated firmware update mechanism
- Completely eliminates raw signal exposure surface

---

## License

MIT — see LICENSE file.

---

## Acknowledgements

- WESAD dataset: Schmidt et al., 2018 —
  "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection"
- Groq for free LLM inference API
- MLflow, FastAPI, PyTorch, MinIO — open source projects
