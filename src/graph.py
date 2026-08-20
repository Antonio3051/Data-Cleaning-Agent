"""LangGraph definition: shared state, node wiring, and conditional routing.

Skeleton only - no implementation yet.
"""

from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import StateGraph


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


class CleaningState(TypedDict):
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
    applied_log: Annotated[list[str], lambda a, b: a + b]
    awaiting_user: bool


def build_graph(checkpointer: Any | None = None) -> Any:
    """Wire nodes/edges and compile the graph.

    Interrupts before ``apply_decisions`` so Streamlit can collect approvals.
    """
    raise NotImplementedError


def route_after_apply(state: CleaningState) -> Literal["profile", "report"]:
    """Loop back for another pass if issues remain, else finish."""
    raise NotImplementedError


__all__ = [
    "CleaningState",
    "ColumnIssue",
    "Decision",
    "PlanStep",
    "StateGraph",
    "build_graph",
    "route_after_apply",
]
