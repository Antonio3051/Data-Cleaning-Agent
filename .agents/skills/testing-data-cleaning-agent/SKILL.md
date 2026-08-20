---
name: testing-data-cleaning-agent
description: How to run and end-to-end test the Streamlit + LangGraph data cleaning agent locally (server startup, Gemini key, golden-path expectations for examples/messy_customers.csv).
---

# Testing the Data Cleaning Agent

## Running the app
```bash
cd <repo>
GEMINI_API_KEY=... .venv/bin/streamlit run app.py --server.headless true --server.port 8501 > /tmp/streamlit.log 2>&1 &
```
- The venv is created by the blueprint (`python3 -m venv .venv`, `pip install -r requirements.txt`).
- The key MUST be exported in the shell that starts Streamlit (`app.py` also loads `.env` via `python-dotenv`).
  Without it the run still completes but the plan caption reads `fallback` instead of
  `gemini:<model>` — always check that caption to know which path you exercised.
- Sanity-check credentials before a long UI run:
  `.venv/bin/python -c "from src.nodes import get_llm; print(get_llm().invoke('say ok').content)"`

## Devin Secrets Needed
- `GEMINI_API_KEY` (reasoning node). `GLM_API_KEY` is legacy and no longer used by `src/nodes.py`.

## UI flow (app.py)
Upload → "Analyze dataset" → section 2 report → section 3 plan cards (per-card "Apply"
checkbox + "Action" selectbox; `fill_constant` reveals a "Constant" text input,
`impute_knn` a "Neighbours" number input) → "Apply approved actions" → section 4 result with
"Download cleaned CSV" and "Clean another dataset".
Unchecking "Apply" disables that card's Action dropdown — that is expected, not a bug.

## Getting expected values without guessing
Precompute ground truth instead of trusting the prompt/issue text:
```bash
.venv/bin/python -c "
import pandas as pd; from src import nodes
d=pd.read_csv('examples/messy_customers.csv')
df=nodes.ingest_node({'original_df':d})['current_df']
print(df.isna().sum(), df.duplicated().sum())"
```
For `examples/messy_customers.csv` this yields 5 columns with nulls (city 8, annual_income 35,
satisfaction 9, churn_risk 6, legacy_segment 127), 7 duplicate rows, 185 missing cells.

## Exercising the human-in-the-loop loop
Decline (uncheck Apply) at least one column that still has nulls: the graph then routes back to
`profile` and shows a **second** review round for that column (`route_after_apply` in
src/graph.py). Approving everything in round 1 finishes in a single pass and does not prove the loop.

## Verifying the download
Streamlit saves to `~/Downloads/cleaned_dataset.csv`; assert with pandas that nulls == 0,
duplicates == 0, dropped columns absent, dates parse, and any override constant you typed is
present in the column you overrode.

## Known noise
`/tmp/streamlit.log` prints repeated `use_container_width` deprecation warnings (Streamlit
removes it after 2025-12-31). Harmless today, but a future Streamlit bump may break the app.
