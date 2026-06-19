import os
from datetime import datetime

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("CANARY_API_URL", "http://localhost:8001")

st.set_page_config(page_title="Canary Deployment Dashboard", layout="wide")

HEALTH_COLOR = {"healthy": "#7ED321", "degraded": "#F5A623", "critical": "#D0021B"}
STATUS_COLOR = {"stable": "#7ED321", "canary_running": "#F5A623",
                "promoting": "#4A90D9", "rolling_back": "#D0021B"}


# ----------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------

def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=10.0)


def api_get(path: str):
    try:
        with _client() as c:
            r = c.get(path)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        st.error(f"API GET {path} failed: {e}")
        return None


def api_post(path: str, json: dict | None = None):
    try:
        with _client() as c:
            r = c.post(path, json=json or {})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"{e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"API POST {path} failed: {e}")
        return None


def api_patch(path: str, json: dict):
    try:
        with _client() as c:
            r = c.patch(path, json=json)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"{e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"API PATCH {path} failed: {e}")
        return None


def list_deployment_names() -> list[str]:
    data = api_get("/deployments")
    if not data:
        return []
    return [d["name"] for d in data.get("deployments", [])]


def _status_badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#888")
    return (f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:6px;font-weight:600'>{status.upper().replace('_',' ')}</span>")


def _health_badge(status: str) -> str:
    color = HEALTH_COLOR.get(status, "#888")
    return (f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:6px;font-weight:600'>{status.upper()}</span>")


# ----------------------------------------------------------------------
# Page 1: Live Traffic Split
# ----------------------------------------------------------------------

def page_live_traffic():
    st.header("Live Traffic Split")
    names = list_deployment_names()
    if not names:
        st.info("No deployments found. Create one on the 'Deploy New Canary' page.")
        return

    col_sel, col_refresh = st.columns([4, 1])
    name = col_sel.selectbox("Deployment", names, key="live_dep")
    if col_refresh.button("Refresh", width="stretch"):
        st.rerun()

    d = api_get(f"/deployments/{name}")
    if not d:
        return

    st.markdown(f"**Status:** {_status_badge(d['status'])}", unsafe_allow_html=True)

    canary_pct = d["canary_traffic_pct"]
    baseline_pct = 100.0 - canary_pct

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Baseline")
        b = d.get("baseline") or {}
        st.metric("Model", f"{b.get('name','-')} v{b.get('version','-')}")
        st.metric("Accuracy", f"{(b.get('metrics') or {}).get('accuracy', float('nan')):.3f}"
                  if (b.get("metrics") or {}).get("accuracy") is not None else "-")
        st.metric("Traffic", f"{baseline_pct:.0f}%")
    with c2:
        st.subheader("Canary")
        cn = d.get("canary")
        if cn:
            st.metric("Model", f"{cn.get('name','-')} v{cn.get('version','-')}")
            st.metric("Accuracy", f"{(cn.get('metrics') or {}).get('accuracy', float('nan')):.3f}"
                      if (cn.get("metrics") or {}).get("accuracy") is not None else "-")
            st.metric("Traffic", f"{canary_pct:.0f}%")
        else:
            st.write("_No canary running_")
    with c3:
        st.subheader("Traffic Split")
        split_df = pd.DataFrame({
            "role": ["baseline", "canary"],
            "pct": [baseline_pct, canary_pct],
        })
        donut = (
            alt.Chart(split_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta="pct:Q",
                color=alt.Color("role:N", scale=alt.Scale(
                    domain=["baseline", "canary"], range=["#7ED321", "#F5A623"])),
                tooltip=["role", "pct"],
            )
        )
        st.altair_chart(donut, width="stretch")

    st.divider()
    st.subheader("Actions")
    a1, a2, a3, a4 = st.columns(4)
    running = d["status"] == "canary_running"
    if a1.button("Promote", disabled=not running, width="stretch"):
        if api_post(f"/deployments/{name}/promote", {"reason": "manual (UI)"}):
            st.success("Promoted."); st.rerun()
    if a2.button("Rollback", disabled=not running, width="stretch"):
        if api_post(f"/deployments/{name}/rollback", {"reason": "manual (UI)"}):
            st.warning("Rolled back."); st.rerun()
    if a3.button("+10% Traffic", disabled=not running, width="stretch"):
        new_pct = min(100.0, canary_pct + 10.0)
        if api_patch(f"/deployments/{name}/traffic", {"canary_traffic_pct": new_pct}):
            st.rerun()
    if a4.button("-10% Traffic", disabled=not running, width="stretch"):
        new_pct = max(0.0, canary_pct - 10.0)
        if api_patch(f"/deployments/{name}/traffic", {"canary_traffic_pct": new_pct}):
            st.rerun()


# ----------------------------------------------------------------------
# Page 2: Model Comparison
# ----------------------------------------------------------------------

def page_comparison():
    st.header("Model Comparison")
    names = list_deployment_names()
    if not names:
        st.info("No deployments found.")
        return
    name = st.selectbox("Deployment", names, key="cmp_dep")

    health = api_get(f"/deployments/{name}/health")
    if not health:
        return
    latest = health.get("latest")
    if not latest:
        st.info("No health snapshots yet — send some prediction traffic first.")
        return

    b = latest["baseline_stats"]
    c = latest["canary_stats"]

    def delta_row(label, bv, cv, fmt, lower_better=True, pct=False):
        delta = cv - bv
        better = delta <= 0 if lower_better else delta >= 0
        arrow = "🟢" if better else "🔴"
        if pct:
            return {"Metric": label, "Baseline": f"{bv:.2%}", "Canary": f"{cv:.2%}",
                    "Delta": f"{arrow} {delta:+.2%}"}
        return {"Metric": label, "Baseline": fmt.format(bv), "Canary": fmt.format(cv),
                "Delta": f"{arrow} {delta:+.1f}"}

    rows = [
        delta_row("Error Rate", b["error_rate"], c["error_rate"], "{:.2%}", pct=True),
        delta_row("Latency P50 (ms)", b["latency_p50_ms"], c["latency_p50_ms"], "{:.1f}"),
        delta_row("Latency P95 (ms)", b["latency_p95_ms"], c["latency_p95_ms"], "{:.1f}"),
        {"Metric": "Requests (5 min)", "Baseline": str(b["request_count"]),
         "Canary": str(c["request_count"]), "Delta": "-"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    history = health.get("history", [])
    if len(history) >= 2:
        recs = []
        for h in reversed(history):  # chronological
            ts = h["checked_at"]
            recs.append({"time": ts, "metric": "baseline_error", "value": h["baseline_stats"]["error_rate"]})
            recs.append({"time": ts, "metric": "canary_error", "value": h["canary_stats"]["error_rate"]})
        edf = pd.DataFrame(recs)
        st.subheader("Error Rate Over Time")
        chart = alt.Chart(edf).mark_line(point=True).encode(
            x="time:T", y="value:Q", color="metric:N")
        st.altair_chart(chart, width="stretch")

        lrecs = []
        for h in reversed(history):
            ts = h["checked_at"]
            lrecs.append({"time": ts, "metric": "baseline_p95", "value": h["baseline_stats"]["latency_p95_ms"]})
            lrecs.append({"time": ts, "metric": "canary_p95", "value": h["canary_stats"]["latency_p95_ms"]})
        ldf = pd.DataFrame(lrecs)
        st.subheader("Latency P95 Over Time")
        lchart = alt.Chart(ldf).mark_line(point=True).encode(
            x="time:T", y="value:Q", color="metric:N")
        st.altair_chart(lchart, width="stretch")
    else:
        st.caption("Need at least 2 health snapshots to draw trend charts.")


# ----------------------------------------------------------------------
# Page 3: Deployment History
# ----------------------------------------------------------------------

EVENT_ICON = {
    "canary_started": "🟢", "traffic_adjusted": "📊", "promoted": "✅",
    "rolled_back": "⏪", "health_check_passed": "✔️", "health_check_failed": "❌",
    "auto_rollback_triggered": "🔴",
}


def page_history():
    st.header("Deployment History")
    data = api_get("/deployments")
    if not data or not data.get("deployments"):
        st.info("No deployments found.")
        return

    rows = []
    for d in data["deployments"]:
        rows.append({
            "Name": d["name"],
            "Baseline": f"{(d.get('baseline') or {}).get('name','-')} v{(d.get('baseline') or {}).get('version','-')}",
            "Status": d["status"],
            "Canary Traffic": f"{d['canary_traffic_pct']:.0f}%",
            "Started": d["started_at"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.divider()
    name = st.selectbox("Inspect event timeline", [d["name"] for d in data["deployments"]])
    events = api_get(f"/deployments/{name}/events")
    if not events or not events.get("events"):
        st.info("No events recorded.")
        return

    st.subheader(f"Event Timeline — {name}")
    for e in events["events"]:  # API returns newest first
        icon = EVENT_ICON.get(e["event_type"], "•")
        detail_bits = []
        for k in ("reason", "recommendation", "health_status", "new_canary_traffic_pct"):
            if k in e["details"]:
                detail_bits.append(f"{k}={e['details'][k]}")
        st.markdown(
            f"{icon} **{e['event_type']}** · {e['canary_traffic_pct']:.0f}% canary "
            f"· `{e['created_at']}`  \n<span style='color:#888'>{', '.join(detail_bits)}</span>",
            unsafe_allow_html=True,
        )

    health = api_get(f"/deployments/{name}/health")
    if health and len(health.get("history", [])) >= 1:
        hrecs = []
        for h in reversed(health["history"]):
            hrecs.append({"time": h["checked_at"],
                          "status": {"healthy": 2, "degraded": 1, "critical": 0}.get(h["health_status"], 2),
                          "label": h["health_status"]})
        hdf = pd.DataFrame(hrecs)
        st.subheader("Health Status Over Canary Lifetime")
        chart = alt.Chart(hdf).mark_line(point=True).encode(
            x="time:T",
            y=alt.Y("status:Q", scale=alt.Scale(domain=[0, 2]),
                    axis=alt.Axis(values=[0, 1, 2], labelExpr="datum.value == 2 ? 'healthy' : datum.value == 1 ? 'degraded' : 'critical'")),
            tooltip=["time", "label"],
        )
        st.altair_chart(chart, width="stretch")


# ----------------------------------------------------------------------
# Page 4: Deploy New Canary
# ----------------------------------------------------------------------

def page_deploy():
    st.header("Deploy New Canary")

    models = api_get("/models")
    model_names = models.get("models", []) if models else []
    deployments = list_deployment_names()

    if not deployments:
        st.info("No deployments exist yet. Create one with the CLI: `canary deploy <name> <model>`.")
    if not model_names:
        st.warning("No models registered yet.")
        return

    with st.form("start_canary"):
        deployment_name = st.selectbox("Deployment", deployments) if deployments else st.text_input("Deployment name")
        canary_model = st.selectbox("Canary model", model_names)
        versions_data = api_get(f"/models/{canary_model}/versions") if canary_model else None
        version_nums = [v["version"] for v in (versions_data.get("versions", []) if versions_data else [])]
        canary_version = st.selectbox("Canary version", version_nums) if version_nums else None
        traffic = st.slider("Initial traffic %", min_value=5, max_value=50, value=10, step=5)
        auto_rollback = st.checkbox("Auto-rollback enabled", value=True)
        submitted = st.form_submit_button("Start Canary Deployment", width="stretch")

    if submitted:
        if not deployment_name:
            st.error("Deployment name is required.")
            return
        payload = {
            "canary_model_name": canary_model,
            "canary_model_version": canary_version,
            "initial_traffic_pct": float(traffic),
            "auto_rollback_enabled": auto_rollback,
        }
        result = api_post(f"/deployments/{deployment_name}/canary", payload)
        if result:
            st.success(f"Canary started on '{deployment_name}' at {traffic}%.")
            st.json(result)


# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------

PAGES = {
    "Live Traffic Split": page_live_traffic,
    "Model Comparison": page_comparison,
    "Deployment History": page_history,
    "Deploy New Canary": page_deploy,
}


def main():
    st.sidebar.title("Canary Deployment")
    st.sidebar.caption(f"API: {API_URL}")
    choice = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
else:
    main()
