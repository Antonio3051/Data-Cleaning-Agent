"""Streamlit UI: upload -> review report -> approve/override plan -> download.

Skeleton only.
"""

import streamlit as st


def render_upload() -> None:
    """CSV/Excel uploader; initializes the graph state and thread id."""
    raise NotImplementedError


def render_missing_report() -> None:
    """Table + chart of the missing-values report."""
    raise NotImplementedError


def render_plan_review() -> None:
    """One approve/override control per PlanStep; writes ``decisions`` into the state."""
    raise NotImplementedError


def render_results() -> None:
    """Before/after preview, applied-actions log, download button."""
    raise NotImplementedError


def main() -> None:
    """Page config, session_state bootstrap, step router."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
