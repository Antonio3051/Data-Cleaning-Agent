"""Streamlit UI for the data cleaning agent.

Upload -> review the missing-values report and distributions -> approve or
override the agent's plan -> apply -> download the cleaned dataset.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import tools
from src.graph import build_graph
from src.nodes import GEMINI_MODEL, ROW_SCOPE, TABLE_ACTIONS

load_dotenv()

ACTION_LABELS = {
    "drop_duplicates": "Drop duplicate rows",
    "impute_mean": "Impute with mean",
    "impute_median": "Impute with median",
    "impute_mode": "Impute with mode",
    "impute_knn": "Impute with KNN",
    "fill_constant": "Fill with a constant",
    "drop_rows": "Drop rows with nulls",
    "drop_column": "Drop the column",
    "keep_as_is": "Keep as is",
}
SEVERITY_COLORS = {"low": "#7dd3a0", "medium": "#f6c667", "high": "#f19b6a", "critical": "#e0685f"}


# --------------------------------------------------------------------------- #
# Graph session plumbing
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_graph() -> Any:
    """One compiled graph (and one in-memory checkpointer) per server process."""
    return build_graph()


def graph_state() -> dict[str, Any]:
    """Current checkpointed values for this Streamlit session's thread."""
    snapshot = get_graph().get_state(st.session_state.config)
    return dict(snapshot.values) if snapshot and snapshot.values else {}


def start_run(df: pd.DataFrame) -> None:
    """Kick off a fresh thread; it pauses at the human-review interrupt."""
    st.session_state.config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    st.session_state.stage = "review"
    with st.spinner(f"Profiling the dataset and asking {GEMINI_MODEL} for a plan..."):
        get_graph().invoke({"original_df": df, "current_df": df, "passes": 0}, st.session_state.config)


def resume_run(decisions: list[dict[str, Any]]) -> None:
    """Write the decisions into the paused state and let the graph continue."""
    graph = get_graph()
    graph.update_state(st.session_state.config, {"decisions": decisions})
    with st.spinner("Applying the approved actions..."):
        graph.invoke(None, st.session_state.config)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def render_upload() -> None:
    """CSV/Excel uploader that seeds the graph state."""
    st.subheader("1. Upload a dataset")
    upload = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    if upload is None:
        st.info("Upload a file to start. The agent profiles it, then proposes a cleaning plan for your approval.")
        return

    df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    st.success(f"Loaded **{upload.name}** — {len(df):,} rows x {df.shape[1]} columns.")
    st.dataframe(df.head(20), use_container_width=True)
    if st.button("Analyze dataset", type="primary"):
        start_run(df)
        st.rerun()


def render_missing_report(state: dict[str, Any]) -> None:
    """Missing-values table, duplicates, and per-column distribution charts."""
    st.subheader("2. Data quality report")
    report = [e for e in state.get("missing_report", []) if e["missing_count"] > 0]
    dupes = state.get("duplicate_report", {})

    cols = st.columns(3)
    cols[0].metric("Rows", f"{len(state['current_df']):,}")
    cols[1].metric("Columns with nulls", len(report))
    cols[2].metric("Duplicate rows", dupes.get("duplicate_rows", 0))

    if state.get("applied_log"):
        with st.expander("Normalizations already applied"):
            for line in state["applied_log"]:
                st.write("-", line)

    if not report:
        st.success("No missing values found.")
        return

    table = pd.DataFrame(
        [
            {
                "Column": e["column"],
                "Type": e["inferred_type"],
                "Missing": e["missing_count"],
                "Missing %": e["missing_pct"],
                "Severity": e["severity"],
                "Mechanism": e["mechanism"],
                "Why": e["mechanism_rationale"],
            }
            for e in report
        ]
    )
    st.dataframe(
        table.style.map(lambda v: f"background-color: {SEVERITY_COLORS.get(v, '')}", subset=["Severity"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Distributions of the affected columns**")
    distributions = state.get("distributions", {})
    for entry in report:
        data = distributions.get(entry["column"], {})
        if data.get("kind") in {None, "empty"}:
            continue
        with st.expander(f"{entry['column']} — {entry['missing_pct']}% missing ({entry['mechanism']})"):
            chart = pd.DataFrame({"count": data["counts"]}, index=data["bin_labels"])
            st.bar_chart(chart)
            if data.get("box"):
                st.caption(
                    "min {min:.2f} | q1 {q1:.2f} | median {median:.2f} | q3 {q3:.2f} | max {max:.2f}".format(**data["box"])
                )
            if data.get("missingness_split"):
                st.caption("Mean of other columns when this one is present vs missing (evidence for MAR):")
                st.dataframe(pd.DataFrame(data["missingness_split"]).T, use_container_width=True)


def render_plan_review(state: dict[str, Any]) -> None:
    """One approve/override control per plan step."""
    st.subheader("3. Review the agent's plan")
    plan = state.get("plan", [])
    if not plan:
        return

    st.caption(f"Proposed by `{state.get('plan_source', 'unknown')}`")
    st.info(state.get("plan_summary", ""))

    decisions: list[dict[str, Any]] = []
    for step in plan:
        with st.container(border=True):
            head, control = st.columns([3, 2])
            title = "Whole table" if step["column"] == ROW_SCOPE else step["column"]
            head.markdown(f"**{title}** — {ACTION_LABELS.get(step['action'], step['action'])}")
            head.caption(f"{step['rationale']}\n\nRisk: {step['risk'] or 'not stated'}")
            head.caption(f"Confidence: {step['confidence']}")

            approved = control.checkbox("Apply", value=True, key=f"ok_{step['step_id']}")
            if step["action"] in TABLE_ACTIONS:
                decisions.append(
                    {"step_id": step["step_id"], "approved": approved, "override_action": None, "override_params": None}
                )
                continue

            options = [a for a in ACTION_LABELS if a not in TABLE_ACTIONS]
            override = control.selectbox(
                "Action",
                options,
                index=options.index(step["action"]) if step["action"] in options else 0,
                format_func=lambda a: ACTION_LABELS[a],
                key=f"act_{step['step_id']}",
                disabled=not approved,
            )
            params = dict(step["params"])
            if override == "fill_constant":
                params = {"value": control.text_input("Constant", value=str(params.get("value", "Unknown")), key=f"c_{step['step_id']}")}
            elif override == "impute_knn":
                params = {"n_neighbors": control.number_input("Neighbours", 1, 50, int(params.get("n_neighbors", 5)), key=f"k_{step['step_id']}")}
            elif override != step["action"]:
                params = {}

            decisions.append(
                {
                    "step_id": step["step_id"],
                    "approved": approved,
                    "override_action": override if override != step["action"] else None,
                    "override_params": params,
                }
            )

    if st.button("Apply approved actions", type="primary"):
        resume_run(decisions)
        st.rerun()


def render_results(state: dict[str, Any]) -> None:
    """Before/after diff, audit log, and the download button."""
    st.subheader("4. Result")
    summary = state["final_report"]
    cols = st.columns(4)
    cols[0].metric("Rows", f"{summary['rows_after']:,}", summary["rows_after"] - summary["rows_before"])
    cols[1].metric("Columns", summary["columns_after"], summary["columns_after"] - summary["columns_before"])
    cols[2].metric("Missing cells", summary["missing_after"], summary["missing_after"] - summary["missing_before"])
    cols[3].metric("Dropped columns", len(summary["dropped_columns"]))

    st.markdown("**Audit trail**")
    for line in state.get("applied_log", []):
        st.write("-", line)

    st.dataframe(state["current_df"].head(50), use_container_width=True)
    st.download_button(
        "Download cleaned CSV",
        state["current_df"].to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        type="primary",
    )
    with st.expander("Report as JSON"):
        st.code(json.dumps(summary, indent=2, default=str), language="json")
    if st.button("Clean another dataset"):
        st.session_state.stage = "upload"
        st.rerun()


def main() -> None:
    """Page config and step routing."""
    st.set_page_config(page_title="Data Cleaning Agent", page_icon="🧹", layout="wide")
    st.title("🧹 Data Cleaning Agent")
    st.caption(f"LangGraph + Pandas, with {GEMINI_MODEL} reasoning over a deterministic missing-values report.")
    st.session_state.setdefault("stage", "upload")

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. **Profile** — Pandas computes the missing-values report, no LLM involved.\n"
            "2. **Plan** — the model proposes one justified action per column.\n"
            "3. **Approve** — nothing touches your data until you say so.\n"
            "4. **Apply** — only registered Pandas functions run."
        )
        st.caption(f"Available actions: {', '.join(tools.ACTION_REGISTRY)}")

    if st.session_state.stage == "upload":
        render_upload()
        return

    state = graph_state()
    if not state:
        st.session_state.stage = "upload"
        st.rerun()

    render_missing_report(state)
    if state.get("final_report"):
        render_results(state)
    else:
        render_plan_review(state)


if __name__ == "__main__":
    main()
