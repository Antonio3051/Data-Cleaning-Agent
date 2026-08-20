# Data Cleaning Agent

A human-in-the-loop data cleaning agent built with **LangGraph**, **Pandas**, **Gemini** and **Streamlit**.

Upload a CSV/Excel file and the agent profiles it with deterministic Pandas code, asks Gemini to justify one
cleaning action per problematic column, waits for your approval, applies only what you approved, and gives you
the cleaned dataset plus an audit trail.

The core safety property: **the model proposes, Pandas mutates, you approve.** The LLM can only emit keys of
`tools.ACTION_REGISTRY`, so it can never touch the data directly or run arbitrary code.

## Graph flow

```
START -> ingest -> profile -> plan -> human_review --(interrupt)--> apply_decisions -> report -> END
                      ^                                                    |
                      +----------------- another pass (max 3) -------------+
```

| Node | Role |
| --- | --- |
| `ingest` | Working copy, header/whitespace normalization, text-to-number and date parsing |
| `profile` | Missing-value counts, severity, missingness correlation, MCAR/MAR/MNAR classification, distribution data — no LLM |
| `plan` | The only LLM call: one justified `PlanStep` per column, with a deterministic rule-based fallback |
| `human_review` | Interrupt point; the UI writes `decisions` into the checkpointed state and resumes the thread |
| `apply_decisions` | Runs the approved (or overridden) registry functions and appends to the audit log |
| `report` | Before/after diff, dtypes, downloadable CSV |

Dropping rows or columns changes the statistics of everything that is left, so the graph re-profiles and
re-plans after each pass, up to three passes.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add your Gemini API key
streamlit run app.py
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required for the reasoning node; without it the deterministic fallback plan is used |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model id |
| `GEMINI_TEMPERATURE` | unset | Only sent when set (some models reject it) |

`examples/messy_customers.csv` is a small, deliberately dirty dataset for a first run.

## Layout

```
app.py          Streamlit UI (upload, report, plan approval, download)
src/graph.py    CleaningState, routing, checkpointing, interrupt
src/nodes.py    Node logic and the Gemini reasoning node
src/tools.py    Pure Pandas tools + ACTION_REGISTRY
```
