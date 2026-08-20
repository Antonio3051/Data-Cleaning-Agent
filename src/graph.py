"""LangGraph definition: shared state, node wiring, and conditional routing.

Flow::

    ingest -> profile -> plan -> human_review -[interrupt]- apply_decisions -> report
                  ^                                              |
                  +----------------- another pass ---------------+

The graph stops after ``human_review``; the Streamlit app writes the user's
``decisions`` into the checkpointed state and resumes the same thread.
"""

from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

MAX_PASSES = 3


class ColumnIssue(TypedDict):
    """The missing-values report entry for one column, produced by ``tools``."""

    column: str
    dtype: str
    inferred_type: str
    missing_count: int
    missing_pct: float
    severity: Literal["none", "low", "medium", "high", "critical"]
    n_unique: int
    missing_correlation: dict[str, float]
    mechanism: Literal["MCAR", "MAR", "MNAR-suspected", "none"]
    mechanism_rationale: str
    stats: dict[str, Any]


class PlanStep(TypedDict):
    """One suggested cleaning action, proposed by the LLM, awaiting approval."""

    step_id: str
    column: str
    action: str  # key of tools.ACTION_REGISTRY
    params: dict[str, Any]
    rationale: str
    confidence: Literal["low", "medium", "high"]
    risk: str


class Decision(TypedDict):
    """The user's verdict on one PlanStep, collected in the Streamlit UI."""

    step_id: str
    approved: bool
    override_action: str | None
    override_params: dict[str, Any] | None


class CleaningState(TypedDict, total=False):
    """State passed between all nodes of the graph."""

    original_df: pd.DataFrame
    current_df: pd.DataFrame
    missing_report: list[ColumnIssue]
    duplicate_report: dict[str, Any]
    distributions: dict[str, dict[str, Any]]
    plan: list[PlanStep]
    plan_summary: str
    plan_source: str
    decisions: list[Decision]
    applied_log: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    final_report: dict[str, Any]
    awaiting_user: bool
    passes: int


def route_after_apply(state: CleaningState) -> Literal["profile", "report"]:
    """Loop back for another pass while issues remain, else finish.

    A second pass matters because dropping a column or rows changes the
    statistics (and the mechanism) of the columns that are left.
    """
    if state.get("passes", 0) >= MAX_PASSES:
        return "report"
    remaining = state["current_df"].isna().any().any()
    return "profile" if remaining else "report"


def build_graph(checkpointer: Any | None = None) -> Any:
    """Wire the nodes and compile the graph.

    A checkpointer is required for the human-in-the-loop interrupt: it stores the
    paused state so the UI can resume the same ``thread_id`` after approval.
    Dataframes are not msgpack-serializable, so the default saver pickles them.
    """
    from . import nodes  # imported here so ``nodes`` can import the state types

    builder = StateGraph(CleaningState)
    builder.add_node("ingest", nodes.ingest_node)
    builder.add_node("profile", nodes.profile_node)
    builder.add_node("plan", nodes.plan_node)
    builder.add_node("human_review", nodes.human_review_node)
    builder.add_node("apply_decisions", nodes.apply_decisions_node)
    builder.add_node("report", nodes.report_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "profile")
    builder.add_edge("profile", "plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"human_review": "human_review", "report": "report"})
    builder.add_edge("human_review", "apply_decisions")
    builder.add_conditional_edges("apply_decisions", route_after_apply, {"profile": "profile", "report": "report"})
    builder.add_edge("report", END)

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True)),
        interrupt_after=["human_review"],
    )


def _route_after_plan(state: CleaningState) -> Literal["human_review", "report"]:
    """Skip the approval step when there is nothing to approve."""
    return "human_review" if state.get("plan") else "report"


__all__ = [
    "MAX_PASSES",
    "CleaningState",
    "ColumnIssue",
    "Decision",
    "PlanStep",
    "build_graph",
    "route_after_apply",
]
