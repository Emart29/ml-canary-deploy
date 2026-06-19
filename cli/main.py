import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as an installed entry point.
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich import box

from db.base import AsyncSessionLocal
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.registry import ModelRegistry
from core.router import TrafficRouter, get_redis_client
from core.deployment import DeploymentEngine
from core.metrics import CanaryMetrics
from db.models import ModelStatusEnum, HealthStatusEnum

console = Console()

API_URL = os.environ.get("CANARY_API_URL", f"http://localhost:8001")

STATUS_STYLE = {
    "stable": "green",
    "canary_running": "yellow",
    "promoting": "cyan",
    "rolling_back": "red",
}
HEALTH_STYLE = {"healthy": "green", "degraded": "yellow", "critical": "red"}
EVENT_ICON = {
    "canary_started": "[green]+[/green]",
    "traffic_adjusted": "[cyan]~[/cyan]",
    "promoted": "[green]^[/green]",
    "rolled_back": "[yellow]v[/yellow]",
    "health_check_passed": "[green]*[/green]",
    "health_check_failed": "[red]x[/red]",
    "auto_rollback_triggered": "[red]![/red]",
}


def async_cmd(f):
    def wrapper(*args, **kwargs):
        asyncio.run(f(*args, **kwargs))
    wrapper.__name__ = f.__name__
    return wrapper


def _enum_val(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def _traffic_bar(canary_pct: float, width: int = 20) -> str:
    canary_blocks = round((canary_pct / 100.0) * width)
    baseline_blocks = width - canary_blocks
    return (
        f"[green]{'#' * baseline_blocks}[/green][yellow]{'#' * canary_blocks}[/yellow]  "
        f"[green]{100 - canary_pct:.0f}% baseline[/green] | [yellow]{canary_pct:.0f}% canary[/yellow]"
    )


@click.group()
def cli():
    """Canary deployment control plane - deploy, observe, promote, roll back."""
    pass


# ----------------------------------------------------------------------
# 1. models
# ----------------------------------------------------------------------

@cli.command("models")
@click.option("--name", default=None, help="Show all versions of a specific model.")
@async_cmd
async def cmd_models(name):
    """List registered models (or all versions of one)."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        if name:
            versions = await meta.list_model_versions(name)
            if not versions:
                console.print(f"[yellow]No versions found for '{name}'.[/yellow]")
                return
            table = Table(title=f"Versions - {name}", box=box.ROUNDED, show_lines=True)
            table.add_column("Version", justify="right", style="cyan bold")
            table.add_column("Framework")
            table.add_column("Accuracy", justify="right")
            table.add_column("F1", justify="right")
            table.add_column("Registered At")
            table.add_column("Description", style="dim")
            for v in versions:
                acc = v.metrics.get("accuracy")
                f1 = v.metrics.get("f1_weighted")
                table.add_row(
                    str(v.version), v.framework,
                    f"{acc:.3f}" if acc is not None else "-",
                    f"{f1:.3f}" if f1 is not None else "-",
                    v.created_at.strftime("%Y-%m-%d %H:%M"),
                    v.description or "",
                )
            console.print(table)
            return

        names = await meta.list_model_names()
        if not names:
            console.print("[yellow]No models registered yet.[/yellow]")
            return
        table = Table(title="Registered Models", box=box.ROUNDED, show_lines=True)
        table.add_column("Name", style="cyan bold")
        table.add_column("Versions", justify="right")
        table.add_column("Latest", justify="right")
        table.add_column("Framework")
        table.add_column("Latest Accuracy", justify="right")
        for n in names:
            versions = await meta.list_model_versions(n)
            latest = versions[-1]
            acc = latest.metrics.get("accuracy")
            table.add_row(
                n, str(len(versions)), f"v{latest.version}", latest.framework,
                f"{acc:.3f}" if acc is not None else "-",
            )
        console.print(table)


# ----------------------------------------------------------------------
# helpers to build the engine
# ----------------------------------------------------------------------

def _build_engine(meta: MetadataStore) -> DeploymentEngine:
    registry = ModelRegistry(meta, BlobStore())
    router = TrafficRouter(get_redis_client())
    return DeploymentEngine(meta, registry, router)


# ----------------------------------------------------------------------
# 2. deploy
# ----------------------------------------------------------------------

@cli.command("deploy")
@click.argument("deployment_name")
@click.argument("model_name")
@click.option("--version", default=None, type=int)
@async_cmd
async def cmd_deploy(deployment_name, model_name, version):
    """Create a stable deployment with a baseline model."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        engine = _build_engine(meta)
        try:
            d = await engine.create_deployment(deployment_name, model_name, version)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
        baseline = await meta.get_model_version(d.baseline_model_id)
    console.print(f"\n[green]Deployment '[bold]{deployment_name}[/bold]' created.[/green]")
    console.print(f"  Baseline model: [cyan]{baseline.name} v{baseline.version}[/cyan]")
    console.print(f"  Status: [{STATUS_STYLE['stable']}]STABLE[/{STATUS_STYLE['stable']}]\n")


# ----------------------------------------------------------------------
# 3. start
# ----------------------------------------------------------------------

@cli.command("start")
@click.argument("deployment_name")
@click.argument("canary_model_name")
@click.option("--version", default=None, type=int)
@click.option("--traffic", default=10.0, type=float, show_default=True)
@async_cmd
async def cmd_start(deployment_name, canary_model_name, version, traffic):
    """Start a canary deployment at the given traffic percentage."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        engine = _build_engine(meta)
        try:
            d = await engine.start_canary(deployment_name, canary_model_name, version, traffic)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
        canary = await meta.get_model_version(d.canary_model_id)
    console.print(f"\n[green]Canary started for '[bold]{deployment_name}[/bold]'.[/green]")
    console.print(f"  Canary model: [yellow]{canary.name} v{canary.version}[/yellow]")
    console.print("  " + _traffic_bar(d.canary_traffic_pct) + "\n")


# ----------------------------------------------------------------------
# 4. status
# ----------------------------------------------------------------------

async def _print_status(meta: MetadataStore, engine: DeploymentEngine, name: str):
    try:
        s = await engine.get_status(name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    d = s["deployment"]
    status = _enum_val(d.status)
    style = STATUS_STYLE.get(status, "white")
    console.print(f"\n[bold]{d.name}[/bold]  [{style}]{status.upper().replace('_',' ')}[/{style}]")
    console.print("  " + _traffic_bar(d.canary_traffic_pct))

    base = s["baseline"]
    console.print(f"  Baseline: [green]{base.name} v{base.version}[/green]"
                  + (f" (acc {base.metrics.get('accuracy'):.3f})" if base.metrics.get("accuracy") is not None else ""))
    if s["canary"]:
        c = s["canary"]
        console.print(f"  Canary:   [yellow]{c.name} v{c.version}[/yellow]"
                      + (f" (acc {c.metrics.get('accuracy'):.3f})" if c.metrics.get("accuracy") is not None else ""))
    health = s["latest_health"]
    if health:
        hs = _enum_val(health.health_status)
        hstyle = HEALTH_STYLE.get(hs, "white")
        console.print(f"  Health:   [{hstyle}]{hs.upper()}[/{hstyle}] "
                      f"(recommendation: {_enum_val(health.recommendation)})")
    console.print()


@cli.command("status")
@click.argument("deployment_name", required=False)
@async_cmd
async def cmd_status(deployment_name):
    """Show live status of one deployment, or all of them."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        engine = _build_engine(meta)
        if deployment_name:
            await _print_status(meta, engine, deployment_name)
        else:
            deployments = await meta.list_deployments()
            if not deployments:
                console.print("[yellow]No deployments yet.[/yellow]")
                return
            for d in deployments:
                await _print_status(meta, engine, d.name)


# ----------------------------------------------------------------------
# 5. traffic
# ----------------------------------------------------------------------

@cli.command("traffic")
@click.argument("deployment_name")
@click.argument("pct", type=float)
@async_cmd
async def cmd_traffic(deployment_name, pct):
    """Adjust canary traffic percentage (0-100)."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        engine = _build_engine(meta)
        try:
            d = await engine.adjust_traffic(deployment_name, pct)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            return
    console.print(f"\n[green]Traffic updated.[/green]")
    console.print("  " + _traffic_bar(d.canary_traffic_pct) + "\n")


# ----------------------------------------------------------------------
# 6. promote
# ----------------------------------------------------------------------

@cli.command("promote")
@click.argument("deployment_name")
@click.option("--reason", default="manual")
@async_cmd
async def cmd_promote(deployment_name, reason):
    """Promote the canary to baseline."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        d = await meta.get_deployment_by_name(deployment_name)
        if d is None:
            console.print(f"[red]Error:[/red] Deployment '{deployment_name}' not found")
            return
        if d.status != ModelStatusEnum.canary_running:
            console.print(f"[red]Error:[/red] No canary running on '{deployment_name}'")
            return
        canary = await meta.get_model_version(d.canary_model_id)
        if not click.confirm(f"Promote canary {canary.name} v{canary.version} to baseline?", default=False):
            console.print("[dim]Aborted.[/dim]")
            return
        engine = _build_engine(meta)
        d = await engine.promote(deployment_name, reason)
        baseline = await meta.get_model_version(d.baseline_model_id)
    console.print(f"\n[green]Promoted.[/green] New baseline: [cyan]{baseline.name} v{baseline.version}[/cyan]\n")


# ----------------------------------------------------------------------
# 7. rollback
# ----------------------------------------------------------------------

@cli.command("rollback")
@click.argument("deployment_name")
@click.option("--reason", default="manual")
@async_cmd
async def cmd_rollback(deployment_name, reason):
    """Roll back the canary; baseline continues."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        d = await meta.get_deployment_by_name(deployment_name)
        if d is None:
            console.print(f"[red]Error:[/red] Deployment '{deployment_name}' not found")
            return
        if d.status != ModelStatusEnum.canary_running:
            console.print(f"[red]Error:[/red] No canary running on '{deployment_name}'")
            return
        canary = await meta.get_model_version(d.canary_model_id)
        if not click.confirm(f"Roll back canary {canary.name} v{canary.version}? Baseline will continue.", default=False):
            console.print("[dim]Aborted.[/dim]")
            return
        engine = _build_engine(meta)
        await engine.rollback(deployment_name, reason)
    console.print(f"\n[yellow]Rolled back.[/yellow] Canary removed; baseline continues.\n")


# ----------------------------------------------------------------------
# 8. history
# ----------------------------------------------------------------------

@cli.command("history")
@click.argument("deployment_name")
@click.option("--limit", default=20, show_default=True)
@async_cmd
async def cmd_history(deployment_name, limit):
    """Show deployment event history."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        d = await meta.get_deployment_by_name(deployment_name)
        if d is None:
            console.print(f"[red]Error:[/red] Deployment '{deployment_name}' not found")
            return
        events = await meta.get_events(d.id, limit=limit)
    if not events:
        console.print("[yellow]No events recorded.[/yellow]")
        return
    table = Table(title=f"Event History - {deployment_name}", box=box.ROUNDED, show_lines=True)
    table.add_column("Time", style="dim")
    table.add_column("Event")
    table.add_column("Traffic %", justify="right")
    table.add_column("Details", style="dim")
    for e in reversed(events):  # chronological
        et = _enum_val(e.event_type)
        icon = EVENT_ICON.get(et, " ")
        detail_bits = []
        for k in ("reason", "recommendation", "health_status", "new_canary_traffic_pct", "canary_version", "promoted_version", "rolled_back_version"):
            if k in e.details:
                detail_bits.append(f"{k}={e.details[k]}")
        table.add_row(
            e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            f"{icon} {et}",
            f"{e.canary_traffic_pct:.0f}",
            ", ".join(detail_bits),
        )
    console.print(table)


# ----------------------------------------------------------------------
# 9. health
# ----------------------------------------------------------------------

def _delta_cell(canary_v: float, baseline_v: float, lower_is_better: bool = True, suffix: str = "") -> str:
    delta = canary_v - baseline_v
    better = (delta < 0) if lower_is_better else (delta > 0)
    style = "green" if better or abs(delta) < 1e-9 else "red"
    sign = "+" if delta >= 0 else ""
    return f"[{style}]{sign}{delta:.3g}{suffix}[/{style}]"


@cli.command("health")
@click.argument("deployment_name")
@async_cmd
async def cmd_health(deployment_name):
    """Show the latest health snapshot (baseline vs canary)."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        d = await meta.get_deployment_by_name(deployment_name)
        if d is None:
            console.print(f"[red]Error:[/red] Deployment '{deployment_name}' not found")
            return
        h = await meta.get_latest_health(d.id)
    if h is None:
        console.print("[yellow]No health snapshot yet.[/yellow]")
        return
    hs = _enum_val(h.health_status)
    style = HEALTH_STYLE.get(hs, "white")
    console.print(f"\n[bold]Health - {deployment_name}[/bold]   [{style}]{hs.upper()}[/{style}]")
    console.print(f"  Recommendation: {_enum_val(h.recommendation)}\n")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Canary", justify="right")
    table.add_column("Delta", justify="right")
    table.add_row("Error rate",
                  f"{h.baseline_error_rate:.2%}", f"{h.canary_error_rate:.2%}",
                  _delta_cell(h.canary_error_rate, h.baseline_error_rate))
    table.add_row("Latency P50 (ms)",
                  f"{h.baseline_latency_p50_ms:.1f}", f"{h.canary_latency_p50_ms:.1f}",
                  _delta_cell(h.canary_latency_p50_ms, h.baseline_latency_p50_ms))
    table.add_row("Latency P95 (ms)",
                  f"{h.baseline_latency_p95_ms:.1f}", f"{h.canary_latency_p95_ms:.1f}",
                  _delta_cell(h.canary_latency_p95_ms, h.baseline_latency_p95_ms))
    table.add_row("Requests (5 min)",
                  str(h.baseline_request_count), str(h.canary_request_count), "-")
    console.print(table)
    console.print()


# ----------------------------------------------------------------------
# 10. metrics (live from the /metrics endpoint)
# ----------------------------------------------------------------------

@cli.command("metrics")
@click.argument("deployment_name")
@async_cmd
async def cmd_metrics(deployment_name):
    """Live metrics for a deployment, scraped from the API /metrics endpoint."""
    from prometheus_client.parser import text_string_to_metric_families

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{API_URL}/metrics")
            resp.raise_for_status()
            text = resp.text
    except Exception as e:
        console.print(f"[red]Could not reach API at {API_URL}/metrics:[/red] {e}")
        return

    roles = {"baseline": {"req": 0.0, "err": 0.0, "buckets": {}, "lat_count": 0.0},
             "canary": {"req": 0.0, "err": 0.0, "buckets": {}, "lat_count": 0.0}}
    traffic_pct = 0.0

    for fam in text_string_to_metric_families(text):
        for sample in fam.samples:
            lbl = sample.labels
            if lbl.get("deployment") != deployment_name:
                continue
            role = lbl.get("model_role")
            if sample.name == "prediction_requests_total" and role in roles:
                roles[role]["req"] += sample.value
                if lbl.get("status") == "error":
                    roles[role]["err"] += sample.value
            elif sample.name == "prediction_latency_ms_bucket" and role in roles:
                le = float(lbl.get("le", "inf"))
                roles[role]["buckets"][le] = roles[role]["buckets"].get(le, 0.0) + sample.value
            elif sample.name == "prediction_latency_ms_count" and role in roles:
                roles[role]["lat_count"] += sample.value
            elif sample.name == "canary_traffic_pct":
                traffic_pct = sample.value

    def stats(r):
        req = r["req"]
        err_rate = (r["err"] / req) if req > 0 else 0.0
        buckets = sorted(r["buckets"].items())
        p95 = CanaryMetrics._percentile_from_buckets(buckets, r["lat_count"], 0.95)
        return int(req), err_rate, p95

    b_req, b_err, b_p95 = stats(roles["baseline"])
    c_req, c_err, c_p95 = stats(roles["canary"])

    console.print(f"\n[bold]Live Metrics - {deployment_name}[/bold]  "
                  f"(canary traffic {traffic_pct:.0f}%)\n")
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Canary", justify="right")
    table.add_row("Requests", str(b_req), str(c_req))
    table.add_row("Error rate", f"{b_err:.2%}", f"{c_err:.2%}")
    table.add_row("Latency P95 (ms)", f"{b_p95:.1f}", f"{c_p95:.1f}")
    console.print(table)
    console.print()


if __name__ == "__main__":
    cli()
