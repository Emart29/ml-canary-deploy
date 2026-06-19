# ML Canary Deploy

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-traffic_split-red?logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Deploy ML model updates to a fraction of traffic, watch Prometheus metrics for both versions, and automatically promote or roll back — no human required.

## The Problem

When you ship a new version of a machine-learning model, it can quietly start making *worse* predictions — and you often don't find out until customers complain or a report looks off. Swapping the old model for the new one all at once is a gamble.

## What This Does

This tool de-risks that swap. Instead of replacing the old model outright, it sends only a small slice of live traffic (say 10%) to the new "canary" model and keeps the rest on the proven one. It continuously compares the two — how often each makes errors and how fast each responds — and then acts on its own:

- if the new model looks **worse**, it **automatically rolls back** to the old one (no human, no 2 a.m. page),
- if the new model proves **healthy** over enough traffic, it can **automatically promote** it to be the new default,
- and a live dashboard shows the traffic split, the head-to-head metrics, and the full history of what happened and why.

Think of it as a careful, automatic understudy swap: the new performer only gets the full stage once it's proven it won't drop the ball.

## Tech Stack

| Layer | Tool |
| --- | --- |
| Serving + REST API | FastAPI + Uvicorn |
| Traffic split config | Redis (instant, no restart) |
| Metadata store | PostgreSQL + SQLAlchemy (async) |
| Model artifacts | MinIO |
| Metrics | Prometheus (`prometheus_client`) |
| Health / auto-decision | Background asyncio scheduler |
| Dashboard | Streamlit + Altair |
| CLI | Click + Rich |
| sklearn integration | `RegistryPipeline` + `@register_on_fit` |

---

## Architecture

```text
                       Prediction request
                              │
                              ▼
                  POST /predict/{deployment}
                              │
                     ┌────────┴────────┐
                     │  TrafficRouter  │──reads──▶ Redis (canary_traffic_pct)
                     └────────┬────────┘
                  80%         │         20%
                   │          │          │
                   ▼                     ▼
            [ v1 Baseline ]        [ v2 Canary ]
                   │                     │
                   └──── Prometheus ─────┘
                     (requests, errors, latency)
                              │
                              ▼
                  HealthChecker  (every 30s)
                     ┌──────────┴──────────┐
                 CRITICAL?              HEALTHY?
                     │                     │
                     ▼                     ▼
              Auto-rollback        Continue / Auto-promote
                     │
                     ▼
            DeploymentEvent ───▶ PostgreSQL (audit trail)
```

Every prediction is routed by a Redis-backed split, recorded in Prometheus, and
logged to PostgreSQL. A background health checker compares canary vs baseline
error rate and p95 latency and acts on the result automatically.

---

## Services

| Service | URL | Description |
| --- | --- | --- |
| FastAPI | <http://localhost:8001> | Prediction hot path + deployment management |
| Streamlit | <http://localhost:8503> | Live traffic, comparison, history, deploy |
| Prometheus | <http://localhost:9091> | Scrapes `api:8001/metrics` (9091 since infra uses 9090) |

---

## Quick Start — Docker

**Prerequisites:** Docker Desktop running + the shared infrastructure stack.

```bash
# 1. Start shared infrastructure (PostgreSQL + MinIO + Redis)
git clone https://github.com/Emart29/ml-platform-infra
cd ml-platform-infra && docker compose up -d && cd ..

# 2. Clone and start the canary deployer
git clone https://github.com/Emart29/ml-canary-deploy
cd ml-canary-deploy
docker compose up
```

Docker will automatically:

1. Create the database schema (5 tables)
2. Train a strong baseline (v1) and a weak canary (v2) on the Heart Disease dataset
3. Deploy the canary at 20% traffic, send 200 predictions, and trigger an auto-rollback
4. Start the FastAPI on 8001, Streamlit on 8503, and Prometheus on 9091

Open **<http://localhost:8503>** — the dashboard is live.

```bash
docker compose down      # stop
docker compose down -v   # stop and delete data
```

---

## Local Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Emart29/ml-canary-deploy
cd ml-canary-deploy
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
pip install -e .

copy .env.example .env

# Initialize + run the demo
python examples/heart_disease/demo.py

# Start the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001

# Start the dashboard (separate terminal)
streamlit run ui/app.py --server.port 8503
```

---

## CLI

```bash
# List registered models (or all versions of one)
canary models
canary models --name heart_disease_classifier

# Create a stable deployment with a baseline model
canary deploy heart-prod heart_disease_classifier

# Start a canary at 20% traffic
canary start heart-prod heart_disease_classifier --version 2 --traffic 20

# Live status (traffic split bar, health badge)
canary status heart-prod

# Ramp traffic
canary traffic heart-prod 50

# Latest health snapshot (baseline vs canary, coloured deltas)
canary health heart-prod

# Live metrics scraped from the API /metrics endpoint
canary metrics heart-prod

# Promote / roll back (with confirmation)
canary promote heart-prod
canary rollback heart-prod --reason "p95 regression"

# Event history
canary history heart-prod
```

Example `canary status` output:

```text
heart-prod  CANARY RUNNING
  ####################  80% baseline | 20% canary
  Baseline: heart_disease_classifier v1 (acc 0.852)
  Canary:   heart_disease_classifier v2 (acc 0.541)
  Health:   CRITICAL (recommendation: rollback)
```

---

## REST API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/metrics` | Prometheus exposition |
| GET | `/models` | List model names |
| GET | `/models/{name}/versions` | List versions of a model |
| GET | `/models/{name}/versions/{v}` | Version detail |
| GET | `/deployments` | List deployments (optional `?status=`) |
| POST | `/deployments` | Create a stable deployment |
| GET | `/deployments/{name}` | Deployment status |
| POST | `/deployments/{name}/canary` | Start a canary |
| PATCH | `/deployments/{name}/traffic` | Adjust canary traffic % |
| POST | `/deployments/{name}/promote` | Promote canary to baseline |
| POST | `/deployments/{name}/rollback` | Roll back the canary |
| GET | `/deployments/{name}/events` | Event history |
| GET | `/deployments/{name}/health` | Latest health + history |
| POST | `/predict/{deployment_name}` | Route + serve a prediction |

```bash
# Serve a prediction (routed to baseline or canary by the traffic split)
curl -X POST http://localhost:8001/predict/heart-prod \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "patient_42", "features": {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}}'
# -> {"prediction_id": "...", "model_role": "baseline", "prediction": 0.12, "model_version": "v1", "latency_ms": 4.4}
```

---

## Python API

### Register a model

```python
from integrations.sklearn import RegistryPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

pipe = RegistryPipeline(
    steps=[("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())],
    registry=registry, model_name="heart_disease_classifier",
)
await pipe.afit(X_train, y_train, X_val=X_test, y_val=y_test)  # async context
print(f"Registered v{pipe.registered_version.version}")
```

### Start a canary and adjust traffic

```python
from core.deployment import DeploymentEngine

await engine.create_deployment("heart-prod", "heart_disease_classifier", model_version=1)
await engine.start_canary("heart-prod", "heart_disease_classifier", canary_version=2, initial_traffic_pct=10.0)
await engine.adjust_traffic("heart-prod", 25.0)   # ramp up
```

### Let the health checker decide

```python
from core.health import HealthChecker

snapshot = await health_checker.evaluate("heart-prod")
print(snapshot.health_status, snapshot.recommendation)   # critical -> rollback

action = await health_checker.run_auto_decision("heart-prod")   # "rolled_back" | "promoted" | "no_action"
```

---

## sklearn Integration

```python
# Decorator form — wrap any function that returns a fitted estimator
from integrations.sklearn import register_on_fit

@register_on_fit("heart_v2", registry=registry)
def train(X, y):
    return GradientBoostingClassifier(n_estimators=50).fit(X, y)

model, version = await train(X_train, y_train)
```

---

## Prometheus Metrics

| Metric | Type | Labels |
| --- | --- | --- |
| `prediction_requests_total` | Counter | deployment, model_version, model_role, status |
| `prediction_latency_ms` | Histogram | deployment, model_version, model_role |
| `canary_traffic_pct` | Gauge | deployment |
| `model_accuracy` | Gauge | deployment, model_version, model_role |

---

## Auto-Rollback Rules

The health checker compares the canary against the baseline over a 5-minute window:

| Condition | Status | Action |
| --- | --- | --- |
| canary error rate > baseline + `MAX_ERROR_RATE_DELTA` (5%) | CRITICAL | auto-rollback |
| canary p95 > baseline + `MAX_LATENCY_P95_DELTA_MS` (100ms) | CRITICAL | auto-rollback |
| above, but at half the threshold | DEGRADED | warn, hold |
| within thresholds, ≥100 canary requests, canary error ≤ baseline | HEALTHY | auto-promote |
| otherwise | HEALTHY | continue |

---

## Streamlit Dashboard

| Page | Description |
| --- | --- |
| Live Traffic Split | Donut chart, baseline/canary models, Promote / Rollback / ±10% traffic buttons |
| Model Comparison | Baseline-vs-canary metric table with coloured deltas + error-rate and p95 trend charts |
| Deployment History | All deployments + per-deployment event timeline + health-status chart |
| Deploy New Canary | Form to start a canary at a chosen traffic % with auto-rollback toggle |

---

## Project Layout

```text
ml-canary-deploy/
├── core/
│   ├── registry.py     # ModelRegistry — register/load models (MinIO + Postgres)
│   ├── router.py       # TrafficRouter — Redis-backed weighted split
│   ├── deployment.py   # DeploymentEngine — create/start/promote/rollback
│   ├── health.py       # HealthChecker + HealthCheckScheduler
│   └── metrics.py      # CanaryMetrics — Prometheus counters/histograms/gauges
├── store/
│   ├── metadata.py     # async PostgreSQL CRUD
│   └── blob.py         # MinIO wrapper
├── api/
│   ├── main.py         # FastAPI app — predict hot path + management routes
│   └── schemas.py      # Pydantic request/response models
├── integrations/
│   └── sklearn.py      # RegistryPipeline, register_on_fit, evaluate_model
├── cli/main.py         # 10-command Click CLI
├── canary_cli/main.py  # installed `canary` entry point
├── ui/app.py           # 4-page Streamlit dashboard
├── db/
│   ├── models.py       # 5 SQLAlchemy ORM models
│   └── base.py         # async engine (NullPool) + session factory
├── examples/heart_disease/demo.py
├── scripts/docker_init.py
├── prometheus/prometheus.yml
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `POSTGRES_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/ml_platform` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/2` | Redis (db 2 — db 0/1 used by sibling projects) |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `MINIO_BUCKET` | `canary-models` | Bucket for model artifacts |
| `API_PORT` | `8001` | FastAPI port |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Background health-check cadence |
| `AUTO_ROLLBACK_ENABLED` | `true` | Master switch for auto-rollback |
| `MAX_ERROR_RATE_DELTA` | `0.05` | Canary error-rate budget over baseline |
| `MAX_LATENCY_P95_DELTA_MS` | `100.0` | Canary p95 latency budget over baseline |

---

## Part of the ML Platform

This is the third project in a connected MLOps platform:

1. **[ml-feature-store](https://github.com/Emart29/ml-feature-store)** — serves the features that become prediction inputs here.
2. **[pipeline-lineage-tracker](https://github.com/Emart29/pipeline-lineage-tracker)** — records which training data produced each model version.
3. **ml-canary-deploy** (this repo) — safely ships those models to production with automatic rollback.

Together: fast feature serving on the front end, full traceability in the middle, and safe automated deployment at the edge.
