"""End-to-end canary deployment demo on the UCI Heart Disease dataset.

Trains a strong baseline (v1) and an intentionally weak canary (v2), deploys
the canary at 20% traffic, simulates live prediction traffic in-process,
runs a health check that flags the canary CRITICAL, and watches the auto-decision
engine roll it back. ASCII-only output (Windows cp1252 safe).
"""
import asyncio
import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from db.base import AsyncSessionLocal, create_all_tables
from db.models import (
    PredictionLog, HealthSnapshot, DeploymentEvent, Deployment, ModelVersion,
)
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.registry import ModelRegistry
from core.router import TrafficRouter, get_redis_client
from core.deployment import DeploymentEngine
from core.metrics import CanaryMetrics
from core.health import HealthChecker
from integrations.sklearn import RegistryPipeline, evaluate_model
from sqlalchemy import delete

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURE_COLS = COLUMNS[:-1]
DATA_DIR = Path(__file__).parent / "data"
MODEL_NAME = "heart_disease_classifier"
DEPLOYMENT = "heart-prod"


def _load_data() -> pd.DataFrame:
    """Load the Cleveland heart dataset; download once, else synthesize a fallback."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "heart.csv"
    if path.exists():
        return pd.read_csv(path)
    try:
        print("    downloading UCI Cleveland heart dataset...")
        raw = urllib.request.urlopen(UCI_URL, timeout=15).read().decode()
        df = pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS, na_values="?")
        df = df.apply(pd.to_numeric, errors="coerce")
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        df["num"] = (df["num"] > 0).astype(int)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        print(f"    download failed ({e}); generating synthetic fallback")
        rng = np.random.default_rng(42)
        n = 303
        df = pd.DataFrame({c: rng.normal(50, 15, n).round(1) for c in FEATURE_COLS})
        logit = (df["age"] / 60 + df["chol"] / 250 + df["thalach"] / 150 - 2.0)
        df["num"] = (logit + rng.normal(0, 0.5, n) > 0.5).astype(int)
        df.to_csv(path, index=False)
        return df


def _hash_features(features: dict) -> str:
    return hashlib.sha256(json.dumps(features, sort_keys=True, default=str).encode()).hexdigest()


async def _reset(session):
    """Make the demo idempotent - clear prior demo rows in FK-safe order."""
    for model in (PredictionLog, HealthSnapshot, DeploymentEvent, Deployment, ModelVersion):
        await session.execute(delete(model))
    await session.commit()


async def _serve_prediction(meta, registry, router, metrics, model_cache,
                            deployment_name, entity_id, features, force_role=None):
    """In-process mirror of the API's POST /predict path."""
    config = await router.get_deployment_config_by_name(deployment_name, meta)
    role = force_role or router.route(config)
    if role == "canary" and config.get("canary_model_id"):
        model_name, version_num = config["canary_model_name"], int(config["canary_model_version"])
    else:
        role = "baseline"
        model_name, version_num = config["baseline_model_name"], int(config["baseline_model_version"])

    mv = await meta.get_model_version_by_number(model_name, version_num)
    version_str = f"v{mv.version}"
    start = time.perf_counter()
    is_error, err_msg, pred_val, pred_cls, conf = False, None, 0.0, None, None
    try:
        if str(mv.id) not in model_cache:
            model_cache[str(mv.id)], _ = await registry.load(mv.name, mv.version)
        model = model_cache[str(mv.id)]
        X = pd.DataFrame([features])
        proba = model.predict_proba(X)[0]
        pred_cls = int(np.argmax(proba)); conf = float(np.max(proba)); pred_val = float(proba[-1])
    except Exception as exc:
        is_error, err_msg = True, str(exc)[:200]
    latency_ms = (time.perf_counter() - start) * 1000.0
    metrics.record_prediction(deployment_name, version_str, role, latency_ms, is_error)
    await meta.log_prediction(
        deployment_id=config["deployment_id"], model_version_id=mv.id, model_role=role,
        entity_id=entity_id, input_hash=_hash_features(features), prediction=pred_val,
        prediction_class=pred_cls, confidence=conf, latency_ms=latency_ms,
        is_error=is_error, error_message=err_msg,
    )
    return role, is_error


async def run_demo():
    df = _load_data()
    X = df[FEATURE_COLS]
    y = df["num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    metrics = CanaryMetrics()
    model_cache: dict = {}

    async with AsyncSessionLocal() as session:
        await _reset(session)
        meta = MetadataStore(session)
        blob = BlobStore()
        registry = ModelRegistry(meta, blob)
        router = TrafficRouter(get_redis_client())
        engine = DeploymentEngine(meta, registry, router)
        health = HealthChecker(meta, engine, metrics)

        # ---- Step 1: train strong baseline (v1) ----
        print("[1/8] Training baseline model (v1)...")
        v1_pipe = RegistryPipeline(
            steps=[("scaler", StandardScaler()),
                   ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))],
            registry=registry, model_name=MODEL_NAME, description="strong baseline",
        )
        await v1_pipe.afit(X_train, y_train, X_val=X_test, y_val=y_test)
        v1 = v1_pipe.registered_version
        print(f"      {MODEL_NAME} v{v1.version} registered, accuracy={v1.metrics['accuracy']:.3f}")

        # ---- Step 2: train weak canary (v2) ----
        print("[2/8] Training canary model (v2, intentionally weaker)...")
        v2_pipe = RegistryPipeline(
            steps=[("scaler", StandardScaler()),
                   ("clf", GradientBoostingClassifier(
                       n_estimators=3, max_depth=1, learning_rate=0.01, random_state=42))],
            registry=registry, model_name=MODEL_NAME, description="weak canary (underfit)",
        )
        await v2_pipe.afit(X_train, y_train, X_val=X_test, y_val=y_test)
        v2 = v2_pipe.registered_version
        print(f"      {MODEL_NAME} v{v2.version} registered, accuracy={v2.metrics['accuracy']:.3f} (weaker)")

        # ---- Step 3: create deployment ----
        print("[3/8] Creating deployment 'heart-prod' with v1 as baseline...")
        await engine.create_deployment(DEPLOYMENT, MODEL_NAME, model_version=v1.version)
        metrics.set_model_accuracy(DEPLOYMENT, f"v{v1.version}", "baseline", v1.metrics["accuracy"])
        print(f"      deployment created, baseline=v{v1.version}")

        # ---- Step 4: start canary at 20% ----
        print("[4/8] Starting canary v2 at 20% traffic...")
        await engine.start_canary(DEPLOYMENT, MODEL_NAME, canary_version=v2.version, initial_traffic_pct=20.0)
        metrics.set_model_accuracy(DEPLOYMENT, f"v{v2.version}", "canary", v2.metrics["accuracy"])
        print("      canary started: 80% -> v1 (baseline) / 20% -> v2 (canary)")

        # ---- Step 5: simulate live traffic ----
        print("[5/8] Sending 200 prediction requests...")
        test_records = X_test.to_dict(orient="records")
        n_baseline = n_canary = 0
        for i in range(200):
            feats = test_records[i % len(test_records)]
            role, _ = await _serve_prediction(meta, registry, router, metrics, model_cache,
                                               DEPLOYMENT, f"patient_{i}", feats)
            if role == "canary":
                n_canary += 1
            else:
                n_baseline += 1
        # 10 malformed requests forced onto the canary to spike its error rate
        for i in range(10):
            await _serve_prediction(meta, registry, router, metrics, model_cache,
                                    DEPLOYMENT, f"bad_{i}", {"age": 50}, force_role="canary")
        print(f"      sent 200 requests: ~{n_baseline} baseline, ~{n_canary} canary (+10 bad -> canary errors)")

        # ---- Step 6: health check ----
        print("[6/8] Running health check...")
        snap = await health.evaluate(DEPLOYMENT)
        print(f"      Health Status: {snap.health_status.value.upper()}")
        print(f"      Baseline: error_rate={snap.baseline_error_rate:.1%}, p95={snap.baseline_latency_p95_ms:.1f}ms")
        print(f"      Canary:   error_rate={snap.canary_error_rate:.1%}, p95={snap.canary_latency_p95_ms:.1f}ms")
        print(f"      Recommendation: {snap.recommendation.value.upper()}")

        # ---- Step 7: auto-decision ----
        print("[7/8] Running auto-decision...")
        action = await health.run_auto_decision(DEPLOYMENT)
        d = await meta.get_deployment_by_name(DEPLOYMENT)
        print(f"      action={action}; deployment status now: {d.status.value}")
        if action == "rolled_back":
            print("      auto-rollback triggered: v2 removed, v1 continues as baseline")

        # ---- Step 8: event history ----
        print("[8/8] Deployment event history:")
        events = await meta.get_events(d.id, limit=100)
        for e in reversed(events):
            et = e.event_type.value if hasattr(e.event_type, "value") else e.event_type
            print(f"      - {e.created_at.strftime('%H:%M:%S')}  {et}  ({e.canary_traffic_pct:.0f}% canary)")
        print(f"\nDemo complete. {len(events)} events recorded for '{DEPLOYMENT}'.")


async def _main():
    await create_all_tables()
    await run_demo()


if __name__ == "__main__":
    asyncio.run(_main())
