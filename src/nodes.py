"""Node logic. Each node takes ``CleaningState`` and returns a partial state update.

Only ``plan_node`` talks to the LLM: it reads the deterministic report produced by
``tools`` and decides, per column, whether to drop or impute and with which method.
It never sees or edits the dataframe itself.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Literal

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

from . import tools
from .graph import CleaningState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Model configuration - Gemini only
# --------------------------------------------------------------------------- #
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
# Some Gemini models use fixed sampling defaults and reject/ignore temperature, so only send it when asked.
GEMINI_TEMPERATURE = os.getenv("GEMINI_TEMPERATURE")

ImputationAction = Literal[
    "impute_mean",
    "impute_median",
    "impute_mode",
    "impute_knn",
    "fill_constant",
    "drop_rows",
    "drop_column",
    "keep_as_is",
]

# Actions that operate on the whole table instead of a single column.
ROW_SCOPE = "<all rows>"
TABLE_ACTIONS = frozenset({"drop_duplicates"})


class PlanStepModel(BaseModel):
    """One justified decision for one column."""

    column: str = Field(description="Exact column name from the report.")
    action: ImputationAction = Field(description="The cleaning action to apply to this column.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON object of action arguments, e.g. {'n_neighbors': 5} for impute_knn or "
        "{'value': 'Unknown'} for fill_constant. Use an empty object when the action takes no arguments.",
    )
    rationale: str = Field(description="Why this action, citing missing %, skew, outliers and mechanism.")
    confidence: Literal["low", "medium", "high"] = Field(description="How strongly the evidence supports this action.")
    risk: str = Field(default="", description="What this action could distort, e.g. variance shrinkage or lost rows.")

    @field_validator("params", mode="before")
    @classmethod
    def _parse_params(cls, value: Any) -> dict[str, Any]:
        """Gemini serializes open-ended objects as JSON strings, so decode them."""
        if isinstance(value, str):
            try:
                decoded = json.loads(value or "{}")
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return value or {}


class CleaningPlanModel(BaseModel):
    """The full plan returned by the model: one step per column with missing values."""

    steps: list[PlanStepModel]
    overall_summary: str = Field(description="Two or three sentences on the dataset's overall data-quality state.")


SYSTEM_PROMPT = """You are a senior data scientist deciding how to handle missing values.

You receive a deterministic report computed with Pandas. Never invent numbers: reason only \
from the statistics given to you.

For EVERY column in the report that has missing values, output exactly one step, choosing the \
action with these rules:

- missing_pct == 0                      -> keep_as_is
- missing_pct > 60                      -> drop_column, unless the column is clearly a key predictor
- missing_pct < 5 and mechanism = MCAR  -> drop_rows is acceptable (little information lost)
- numeric, |skew| < 0.5, few outliers   -> impute_mean
- numeric, skewed or many outliers      -> impute_median (the mean would be dragged by the tail)
- numeric and mechanism = MAR (missingness correlates with other columns) -> impute_knn, since \
  the observed columns predict the missing values; pass {"n_neighbors": 5}
- categorical, dominant mode            -> impute_mode
- categorical and mechanism = MNAR-suspected -> fill_constant with {"value": "Unknown"}, because \
  the absence itself is informative and must not be hidden
- datetime columns                      -> prefer drop_rows or keep_as_is; never average dates

Each rationale must cite the concrete evidence (missing %, skew value, outlier count, correlated \
column, mechanism). State the risk of the action you chose. Be concise: at most two sentences per field."""


def get_llm(**overrides: Any) -> BaseChatModel:
    """Return the Gemini chat client. This project is configured strictly for Gemini."""
    api_key = os.getenv(GEMINI_API_KEY_ENV) or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing Gemini credentials: set the {GEMINI_API_KEY_ENV} environment variable.")
    params: dict[str, Any] = {
        "model": GEMINI_MODEL,
        "google_api_key": api_key,
        "timeout": 120,
        "max_retries": 2,
    }
    if GEMINI_TEMPERATURE is not None:
        params["temperature"] = float(GEMINI_TEMPERATURE)
    params.update(overrides)
    return ChatGoogleGenerativeAI(**params)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def ingest_node(state: CleaningState) -> dict[str, Any]:
    """Seed the working copy and apply the safe, non-destructive normalizations."""
    df = state["original_df"].copy()
    log: list[str] = []

    df.columns = [str(c).strip() for c in df.columns]
    df, msg = tools.strip_whitespace(df)
    log.append(msg)
    df, msg = tools.unify_case(df)
    log.append(msg)

    for col in df.columns:
        info = tools.detect_column_type(df[col])
        if not info["convertible"]:
            continue
        if info["inferred_type"] == "numeric":
            df, msg = tools.convert_to_numeric(df, col)
            log.append(msg)
        elif info["inferred_type"] == "datetime":
            df, msg = tools.clean_dates(df, col)
            log.append(msg)

    return {"current_df": df, "applied_log": log}


def profile_node(state: CleaningState) -> dict[str, Any]:
    """Deterministic analysis: missing-values report, duplicates and plot data."""
    df = state["current_df"]
    return {
        "missing_report": tools.analyze_missing_values(df),
        "duplicate_report": tools.duplicate_report(df),
        "distributions": tools.distribution_bundle(df),
    }


def plan_node(state: CleaningState) -> dict[str, Any]:
    """Reasoning node: Gemini turns the missing-values report into a justified plan."""
    report = [entry for entry in state["missing_report"] if entry["missing_count"] > 0]
    dedup = _duplicate_step(state)
    if not report:
        return {
            "plan": dedup,
            "plan_summary": "No missing values detected." + (" Duplicate rows are still pending." if dedup else ""),
            "plan_source": "deterministic",
            "awaiting_user": bool(dedup),
        }

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_render_report(state, report))]

    try:
        llm = get_llm().with_structured_output(CleaningPlanModel, method="function_calling")
        result = llm.invoke(messages)
    except Exception:  # never lose the run because of a model/transport error
        logger.exception("Gemini planning call failed; falling back to deterministic rules.")
        return {
            "plan": dedup + [_to_plan_step(s) for s in _fallback_plan(report)],
            "plan_summary": "Gemini was unavailable; this plan comes from the deterministic rule set.",
            "plan_source": "fallback",
            "awaiting_user": True,
        }

    steps = [_to_plan_step(step) for step in result.steps if _column_exists(step.column, state["current_df"])]
    covered = {step["column"] for step in steps}
    steps += [_to_plan_step(s) for s in _fallback_plan(report) if s.column not in covered]
    return {
        "plan": dedup + steps,
        "plan_summary": result.overall_summary,
        "plan_source": f"gemini:{GEMINI_MODEL}",
        "awaiting_user": True,
    }


def _duplicate_step(state: CleaningState) -> list[dict[str, Any]]:
    """Deduplication is clear-cut, so it is proposed by rule rather than by the model."""
    duplicates = int(state.get("duplicate_report", {}).get("duplicate_rows", 0))
    if duplicates <= 0:
        return []
    return [
        {
            "step_id": f"duplicates:{uuid.uuid4().hex[:6]}",
            "column": ROW_SCOPE,
            "action": "drop_duplicates",
            "params": {},
            "rationale": f"{duplicates} fully identical row(s) detected; they bias every statistic downstream.",
            "confidence": "high",
            "risk": "Legitimate repeated observations are lost if the rows are not true duplicates.",
        }
    ]


def _render_report(state: CleaningState, report: list[dict[str, Any]]) -> str:
    """Serialize the deterministic evidence the model must reason over."""
    df = state["current_df"]
    payload = {
        "dataset": {"rows": len(df), "columns": int(df.shape[1]), "column_names": [str(c) for c in df.columns]},
        "duplicates": state.get("duplicate_report", {}),
        "columns": [
            {
                "column": e["column"],
                "inferred_type": e["inferred_type"],
                "missing_count": e["missing_count"],
                "missing_pct": e["missing_pct"],
                "severity": e["severity"],
                "n_unique": e["n_unique"],
                "mechanism": e["mechanism"],
                "mechanism_rationale": e["mechanism_rationale"],
                "missing_correlation": e["missing_correlation"],
                "distribution": e["stats"],
            }
            for e in report
        ],
    }
    return (
        "Missing-values report (computed with Pandas, all figures are exact):\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Return one step per column listed above."
    )


def _column_exists(column: str, df: pd.DataFrame) -> bool:
    if column in df.columns:
        return True
    logger.warning("The model proposed a step for unknown column %r; discarding it.", column)
    return False


def _to_plan_step(step: PlanStepModel) -> dict[str, Any]:
    return {
        "step_id": f"{step.column}:{step.action}:{uuid.uuid4().hex[:6]}",
        "column": step.column,
        "action": step.action,
        "params": step.params,
        "rationale": step.rationale,
        "confidence": step.confidence,
        "risk": step.risk,
    }


def _fallback_plan(report: list[dict[str, Any]]) -> list[PlanStepModel]:
    """Deterministic version of the prompt's rules, used if GLM is unreachable or skips a column."""
    models: list[PlanStepModel] = []
    for entry in report:
        stats = entry.get("stats") or {}
        pct, mechanism = entry["missing_pct"], entry["mechanism"]
        if pct > 60:
            action, params = "drop_column", {}
        elif stats.get("kind") == "numeric":
            if mechanism == "MAR":
                action, params = "impute_knn", {"n_neighbors": 5}
            elif abs(stats.get("skew", 0.0)) >= 0.5 or stats.get("outlier_count", 0) > 0.01 * max(entry["n_unique"], 1):
                action, params = "impute_median", {}
            else:
                action, params = "impute_mean", {}
        elif mechanism == "MNAR-suspected":
            action, params = "fill_constant", {"value": "Unknown"}
        elif pct < 5:
            action, params = "drop_rows", {}
        else:
            action, params = "impute_mode", {}
        models.append(
            PlanStepModel(
                column=entry["column"],
                action=action,
                params=params,
                rationale=f"Rule-based fallback: {pct}% missing, mechanism {mechanism}, stats {stats or 'n/a'}.",
                confidence="low",
                risk="Generated without model reasoning; review before applying.",
            )
        )
    return models


def human_review_node(state: CleaningState) -> dict[str, Any]:
    """Interrupt point: the graph is compiled to stop *after* this node.

    The UI resumes the run once it has written ``decisions`` into the state.
    """
    return {"awaiting_user": True}


def apply_decisions_node(state: CleaningState) -> dict[str, Any]:
    """Execute the approved steps, in order, against ``current_df``."""
    df = state["current_df"]
    decisions = {d["step_id"]: d for d in state.get("decisions", [])}
    log: list[str] = []

    for step in state.get("plan", []):
        decision = decisions.get(step["step_id"])
        if decision is None or not decision["approved"]:
            log.append(f"Skipped '{step['column']}': {step['action']} not approved.")
            continue

        action = decision.get("override_action") or step["action"]
        params = decision.get("override_params")
        if params is None:
            params = step["params"]
        func = tools.ACTION_REGISTRY.get(action)
        if func is None:
            log.append(f"Skipped '{step['column']}': unknown action {action!r}.")
            continue
        table_scope = action in TABLE_ACTIONS
        if not table_scope and step["column"] not in df.columns:
            log.append(f"Skipped '{step['column']}': column no longer present.")
            continue

        try:
            df, message = func(df, **params) if table_scope else func(df, step["column"], **params)
        except Exception as exc:  # a bad param must not kill the whole run
            logger.exception("Action %s failed on column %s", action, step["column"])
            log.append(f"Failed '{step['column']}' ({action}): {exc}")
            continue
        log.append(message)

    return {"current_df": df, "applied_log": log, "decisions": [], "plan": [], "awaiting_user": False, "passes": state.get("passes", 0) + 1}


def report_node(state: CleaningState) -> dict[str, Any]:
    """Before/after diff for the UI, computed from ``original_df`` vs ``current_df``."""
    before, after = state["original_df"], state["current_df"]
    summary = {
        "rows_before": len(before),
        "rows_after": len(after),
        "columns_before": int(before.shape[1]),
        "columns_after": int(after.shape[1]),
        "dropped_columns": [str(c) for c in before.columns if c not in after.columns],
        "missing_before": int(before.isna().sum().sum()),
        "missing_after": int(after.isna().sum().sum()),
        "dtypes_after": {str(c): str(t) for c, t in after.dtypes.items()},
    }
    return {"final_report": summary, "missing_report": tools.analyze_missing_values(after), "awaiting_user": False}
