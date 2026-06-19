import asyncio
import functools
from typing import Any, Callable

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)

from core.registry import ModelRegistry
from db.models import ModelVersion


def evaluate_model(model: Any, X_test, y_test) -> dict:
    """Compute a standard classification metric set. Works for any fitted
    sklearn classifier; roc_auc is only added when it can be computed."""
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    n_classes = len(np.unique(y_true))
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
                )
    except Exception:
        # roc_auc not computable (e.g. a class missing from the eval split) — skip it.
        pass

    return metrics


def _safe_params(pipeline: Pipeline) -> dict:
    """JSON-serialisable snapshot of the pipeline's structure + final estimator params."""
    last_name, last_est = pipeline.steps[-1]
    final_params = {
        k: v for k, v in last_est.get_params().items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    return {
        "steps": [name for name, _ in pipeline.steps],
        "final_estimator": type(last_est).__name__,
        "final_params": final_params,
    }


class RegistryPipeline(Pipeline):
    """sklearn Pipeline that auto-registers the fitted model in the ModelRegistry.

    Sync use (plain scripts/tests):   pipeline.fit(X, y, X_val=Xv, y_val=yv)
    Async use (inside an event loop):  await pipeline.afit(X, y, X_val=Xv, y_val=yv)
    """

    # sklearn validates declared params; mark our extra ctor args as unconstrained.
    _parameter_constraints = {
        **Pipeline._parameter_constraints,
        "registry": [object],
        "model_name": [str],
        "framework": [str],
        "description": [str],
        "tags": [list, None],
    }

    def __init__(
        self, steps, registry: ModelRegistry, model_name: str,
        framework: str = "sklearn", description: str = "",
        tags: list[str] | None = None, *, memory=None, verbose=False,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)
        # Stored verbatim under the ctor arg names — required by sklearn's
        # get_params/_validate_params introspection.
        self.registry = registry
        self.model_name = model_name
        self.framework = framework
        self.description = description
        self.tags = tags or []
        self.registered_version: ModelVersion | None = None

    def _fit_local(self, X, y=None, **fit_params):
        X_val = fit_params.pop("X_val", None)
        y_val = fit_params.pop("y_val", None)
        super().fit(X, y, **fit_params)
        if X_val is not None and y_val is not None:
            metrics = evaluate_model(self, X_val, y_val)
            metrics["n_val_samples"] = int(len(y_val))
        else:
            metrics = evaluate_model(self, X, y)
        try:
            metrics["n_features"] = int(np.asarray(X).shape[1])
        except Exception:
            metrics["n_features"] = None
        metrics["n_samples"] = int(len(y)) if y is not None else None
        self.training_metrics = metrics
        return metrics

    def _register_coro(self):
        # Serialize a clean Pipeline of the already-fitted steps — never `self`,
        # which holds an unpicklable ModelRegistry/DB-session reference.
        clean_model = Pipeline(self.steps)
        return self.registry.register(
            name=self.model_name,
            model_object=clean_model,
            framework=self.framework,
            metrics=self.training_metrics,
            parameters=_safe_params(self),
            description=self.description,
            tags=self.tags,
        )

    def fit(self, X, y=None, **fit_params):
        self._fit_local(X, y, **fit_params)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.registered_version = asyncio.run(self._register_coro())
            return self
        raise RuntimeError(
            "RegistryPipeline.fit() was called inside a running event loop; "
            "use 'await pipeline.afit(...)' instead."
        )

    async def afit(self, X, y=None, **fit_params):
        self._fit_local(X, y, **fit_params)
        self.registered_version = await self._register_coro()
        return self


def register_on_fit(model_name: str, registry: ModelRegistry, **register_kwargs) -> Callable:
    """Decorator for a function that returns a fitted sklearn estimator.
    The wrapped function becomes a coroutine returning (model, ModelVersion)."""

    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            model = fn(*args, **kwargs)
            if asyncio.iscoroutine(model):
                model = await model
            metrics = register_kwargs.pop("metrics", None)
            version = await registry.register(
                name=model_name,
                model_object=model,
                framework=register_kwargs.pop("framework", "sklearn"),
                metrics=metrics or {},
                parameters=register_kwargs.pop("parameters", {}),
                description=register_kwargs.pop("description", ""),
                tags=register_kwargs.pop("tags", []),
            )
            return model, version

        return wrapper

    return decorator
