"""Pandas cleaning primitives, exposed as deterministic tools.

Design rules:
- Every function is pure: it never mutates its input, it returns a new object.
- Nothing here calls an LLM. All statistics the agent reasons about are computed
  by this module, so the model can never hallucinate a number.
- Every mutating function returns ``(new_df, log_message)`` so the caller can
  build an audit trail.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

MissingMechanism = Literal["MCAR", "MAR", "MNAR-suspected", "none"]
Severity = Literal["none", "low", "medium", "high", "critical"]

_THOUSANDS_DOT = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_CURRENCY_NOISE = re.compile(r"[^\d,.\-+eE]")
# Letters other than the exponent 'e' mean the value is a code or a label, not a number.
_ALPHA_NOISE = re.compile(r"[A-DF-Za-df-z]")
_NULL_TOKENS = {"", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?", "unknown", "missing"}

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    # (regex, strptime format, dayfirst)
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "%Y-%m-%d", False),
    (re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"), "%Y/%m/%d", False),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), "%m/%d/%Y", False),  # ambiguous, resolved by _infer_dayfirst
    (re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"), "%m-%d-%Y", False),  # ambiguous, resolved by _infer_dayfirst
    (re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"), "%d.%m.%Y", True),
    (re.compile(r"^\d{8}$"), "%Y%m%d", False),
)

# US month-first defaults, flipped only when the data proves day-first.
_AMBIGUOUS_DAYFIRST = {"%m/%d/%Y": "%d/%m/%Y", "%m-%d-%Y": "%d-%m-%Y"}

# Free-form date parsing is only attempted on values that at least look like a date.
_DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/. ]\d{1,2}[-/. ]\d{1,4}([ T].*)?$|^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$")


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: Literal["first", "last"] = "first",
) -> tuple[pd.DataFrame, str]:
    """Remove duplicate records, optionally keyed on a subset of columns."""
    cols = [c for c in (subset or []) if c in df.columns] or None
    before = len(df)
    out = df.drop_duplicates(subset=cols, keep=keep).reset_index(drop=True)
    removed = before - len(out)
    scope = f"subset={cols}" if cols else "all columns"
    return out, f"Dropped {removed} duplicate row(s) ({scope}, keep={keep})."


def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, Any]:
    """Count duplicates without touching the data."""
    cols = [c for c in (subset or []) if c in df.columns] or None
    mask = df.duplicated(subset=cols, keep="first")
    return {
        "subset": cols,
        "duplicate_rows": int(mask.sum()),
        "duplicate_pct": round(float(mask.mean() * 100), 2) if len(df) else 0.0,
        "example_indices": [int(i) for i in df.index[mask][:5]],
    }


# --------------------------------------------------------------------------- #
# Type detection & format cleaning
# --------------------------------------------------------------------------- #
def _normalize_null_tokens(s: pd.Series) -> pd.Series:
    stripped = s.astype("string").str.strip()
    return stripped.mask(stripped.str.lower().isin(_NULL_TOKENS))


def strip_whitespace(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Trim leading/trailing whitespace, collapse inner runs, and blank out null tokens."""
    out = df.copy()
    targets = columns or [c for c in out.columns if ptypes.is_object_dtype(out[c]) or ptypes.is_string_dtype(out[c])]
    touched: list[str] = []
    for col in targets:
        if col not in out.columns:
            continue
        cleaned = _normalize_null_tokens(out[col]).str.replace(r"\s+", " ", regex=True)
        if not cleaned.equals(out[col].astype("string")):
            touched.append(col)
        out[col] = cleaned
    return out, f"Stripped/normalized whitespace on {len(touched)} column(s): {touched}."


def _case_rank(value: str) -> int:
    """Rank spellings when variants are equally frequent: Title case reads best, SHOUTING worst."""
    if value.istitle():
        return 0
    if value.isupper():
        return 2
    if value.islower():
        return 1
    return 0


def unify_case(df: pd.DataFrame, columns: list[str] | None = None, max_unique: int = 200) -> tuple[pd.DataFrame, str]:
    """Collapse case variants of the same category onto their most frequent spelling.

    ``New York`` / ``new york`` / ``NEW YORK`` all become ``New York`` (the modal
    spelling), so they stop competing as separate categories in counts and mode
    imputation. Values without a case-variant twin are left exactly as written,
    and high-cardinality columns (free text, ids) are skipped.
    """
    out = df.copy()
    targets = columns or [c for c in out.columns if ptypes.is_object_dtype(out[c]) or ptypes.is_string_dtype(out[c])]
    touched: dict[str, int] = {}
    for col in targets:
        if col not in out.columns:
            continue
        series = out[col].astype("string")
        non_null = series.dropna()
        if non_null.empty or non_null.nunique() > max_unique:
            continue
        folded = non_null.str.casefold()
        # The winning spelling per case-insensitive group: most frequent, then the best-looking
        # variant (Title case over ALL CAPS or all lower), then alphabetical for full determinism.
        counts = pd.DataFrame({"folded": folded, "value": non_null}).value_counts(["folded", "value"]).reset_index(name="n")
        counts["shape_rank"] = counts["value"].map(_case_rank)
        canonical = (
            counts.sort_values(["folded", "n", "shape_rank", "value"], ascending=[True, False, True, True])
            .drop_duplicates("folded")
            .set_index("folded")["value"]
        )
        mapped = series.str.casefold().map(canonical).astype("string")
        changed = int((mapped.fillna("") != series.fillna("")).sum())
        if changed:
            touched[col] = changed
            out[col] = mapped
    if not touched:
        return out, "No case-variant categories found."
    detail = ", ".join(f"{col} ({n})" for col, n in touched.items())
    return out, f"Unified letter case on {len(touched)} column(s): {detail}."


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Parse a text series into numbers, handling currency symbols and both decimal conventions."""
    txt = _normalize_null_tokens(s)
    txt = txt.mask(txt.str.match(_DATE_LIKE))  # never read 03/15/2021 as the number 3152021
    txt = txt.str.replace(r"[\s\x00a0]", "", regex=True)
    txt = txt.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (1 234) -> -1234
    txt = txt.mask(txt.str.contains(_ALPHA_NOISE))  # 'A1' is a code, not the number 1
    txt = txt.str.replace(_CURRENCY_NOISE, "", regex=True)

    sample = txt.dropna()
    if sample.empty:
        return pd.to_numeric(txt, errors="coerce")

    comma_decimal = sample.str.match(_THOUSANDS_DOT).mean() > 0.5 or (
        sample.str.contains(",").mean() > 0.5 and not sample.str.match(_THOUSANDS_COMMA).any()
    )
    if comma_decimal:
        txt = txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        txt = txt.str.replace(",", "", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def detect_column_type(s: pd.Series, sample_size: int = 1000) -> dict[str, Any]:
    """Infer the semantic type of a column: numeric, datetime, boolean, categorical or text."""
    non_null = s.dropna()
    result: dict[str, Any] = {
        "pandas_dtype": str(s.dtype),
        "inferred_type": "empty",
        "convertible": False,
        "parse_rate": 0.0,
        "date_format": None,
        "dayfirst": False,
        "n_unique": int(non_null.nunique()),
    }
    if non_null.empty:
        return result
    if ptypes.is_bool_dtype(s):
        return {**result, "inferred_type": "boolean", "parse_rate": 1.0}
    if ptypes.is_numeric_dtype(s):
        return {**result, "inferred_type": "numeric", "parse_rate": 1.0}
    if ptypes.is_datetime64_any_dtype(s):
        return {**result, "inferred_type": "datetime", "parse_rate": 1.0}

    sample = non_null.sample(min(sample_size, len(non_null)), random_state=0)
    as_text = sample.astype("string").str.strip()

    if as_text.str.lower().isin({"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"}).mean() > 0.95:
        return {**result, "inferred_type": "boolean", "convertible": True, "parse_rate": 1.0}

    num_rate = float(_to_numeric_series(sample).notna().mean())
    fmt, dayfirst, date_rate = _detect_date_format(as_text)

    if date_rate >= 0.9 and (fmt is not None or date_rate > num_rate):
        return {
            **result,
            "inferred_type": "datetime",
            "convertible": True,
            "parse_rate": round(date_rate, 3),
            "date_format": fmt,
            "dayfirst": dayfirst,
        }
    if num_rate >= 0.9:
        return {**result, "inferred_type": "numeric", "convertible": True, "parse_rate": round(num_rate, 3)}

    is_categorical = result["n_unique"] <= max(20, int(0.05 * len(non_null)))
    return {**result, "inferred_type": "categorical" if is_categorical else "text", "parse_rate": 0.0}


def convert_to_numeric(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Text-to-number conversion; unparseable entries become NaN."""
    out = df.copy()
    parsed = _to_numeric_series(out[column])
    failures = int(parsed.isna().sum() - out[column].isna().sum())
    out[column] = parsed
    return out, f"Converted '{column}' to numeric ({max(failures, 0)} value(s) unparseable -> NaN)."


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _infer_dayfirst(text: pd.Series) -> bool:
    """Decide US (MM/DD) vs European (DD/MM) when the separator pattern is ambiguous.

    If any first component exceeds 12 the column must be day-first; if any second
    component exceeds 12 it must be month-first. Default to US convention.
    """
    parts = text.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{4}$").dropna()
    if parts.empty:
        return False
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    return bool((first > 12).any() and not (second > 12).any())


def _detect_date_format(text: pd.Series) -> tuple[str | None, bool, float]:
    """Return (strptime format, dayfirst, parse success rate) for the best matching pattern."""
    best: tuple[str | None, bool, float] = (None, False, 0.0)
    for pattern, fmt, dayfirst in _DATE_PATTERNS:
        rate = float(text.str.match(pattern).mean())
        if rate > best[2]:
            if fmt in _AMBIGUOUS_DAYFIRST and _infer_dayfirst(text):
                fmt, dayfirst = _AMBIGUOUS_DAYFIRST[fmt], True
            best = (fmt, dayfirst, rate)
    if best[2] < 0.9 and float(text.str.match(_DATE_LIKE).mean()) >= 0.9:
        loose = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=_infer_dayfirst(text))
        loose_rate = float(loose.notna().mean())
        if loose_rate > best[2]:
            return None, _infer_dayfirst(text), loose_rate
    return best


def clean_dates(df: pd.DataFrame, column: str, output_format: str | None = None) -> tuple[pd.DataFrame, str]:
    """Parse a date column, auto-detecting US vs European order and -, / or . separators."""
    out = df.copy()
    text = _normalize_null_tokens(out[column])
    fmt, dayfirst, rate = _detect_date_format(text)
    if fmt:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        unparsed = parsed.isna() & text.notna()
        if unparsed.any():  # mixed separators inside one column
            parsed = parsed.fillna(pd.to_datetime(text[unparsed], errors="coerce", dayfirst=dayfirst, format="mixed"))
    else:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst, format="mixed")
    out[column] = parsed.dt.strftime(output_format) if output_format else parsed
    detected = fmt or ("day-first mixed" if dayfirst else "month-first mixed")
    return out, f"Parsed '{column}' as dates (format={detected}, success={rate:.0%})."


# --------------------------------------------------------------------------- #
# Missing values analysis
# --------------------------------------------------------------------------- #
def _severity(pct: float) -> Severity:
    if pct == 0:
        return "none"
    if pct < 5:
        return "low"
    if pct < 20:
        return "medium"
    if pct < 50:
        return "high"
    return "critical"


def _classify_mechanism(df: pd.DataFrame, column: str, corr: dict[str, float], threshold: float = 0.2) -> tuple[MissingMechanism, str]:
    """Theoretical classification of the missingness mechanism.

    This is a heuristic, not a proof: MCAR cannot be confirmed from data alone,
    and MNAR is unfalsifiable without the missing values themselves.
    """
    if df[column].isna().sum() == 0:
        return "none", "No missing values."
    strong = {k: v for k, v in corr.items() if abs(v) >= threshold}
    if strong:
        top = max(strong, key=lambda k: abs(strong[k]))
        return "MAR", f"Missingness correlates with '{top}' (r={strong[top]:.2f}), so it is predictable from observed data."
    if df[column].isna().mean() > 0.5:
        return "MNAR-suspected", "Over half the column is missing with no observed predictor; absence may itself carry information."
    return "MCAR", "No association with other columns detected; consistent with missing completely at random."


def missingness_correlation(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Correlate this column's null indicator against every other column.

    Numeric columns use point-biserial (Pearson on the indicator); non-numeric
    columns are compared via their own null indicator.
    """
    indicator = df[column].isna().astype(float)
    if indicator.nunique() < 2:
        return {}
    out: dict[str, float] = {}
    for other in df.columns:
        if other == column:
            continue
        series = df[other]
        candidate = series.astype(float) if ptypes.is_numeric_dtype(series) else series.isna().astype(float)
        if candidate.nunique(dropna=True) < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):  # zero-variance overlap yields NaN, which we drop below
            r = indicator.corr(candidate)
        if pd.notna(r) and abs(r) >= 0.05:
            out[other] = round(float(r), 3)
    return dict(sorted(out.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])


def analyze_missing_values(df: pd.DataFrame, correlation_threshold: float = 0.2) -> list[dict[str, Any]]:
    """Full per-column missing-values report: counts, type, distribution stats, correlation, mechanism."""
    n_rows = len(df)
    report: list[dict[str, Any]] = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct = round(n_missing / n_rows * 100, 2) if n_rows else 0.0
        type_info = detect_column_type(df[col])
        corr = missingness_correlation(df, col) if n_missing else {}
        mechanism, rationale = _classify_mechanism(df, col, corr, correlation_threshold)
        entry: dict[str, Any] = {
            "column": col,
            "dtype": str(df[col].dtype),
            "inferred_type": type_info["inferred_type"],
            "missing_count": n_missing,
            "missing_pct": pct,
            "severity": _severity(pct),
            "n_unique": type_info["n_unique"],
            "missing_correlation": corr,
            "mechanism": mechanism,
            "mechanism_rationale": rationale,
            "stats": column_statistics(df[col]),
        }
        report.append(entry)
    return report


def column_statistics(s: pd.Series) -> dict[str, Any]:
    """Distribution summary used by the LLM to choose mean vs median vs mode."""
    non_null = s.dropna()
    if non_null.empty:
        return {}
    if ptypes.is_numeric_dtype(non_null):
        desc = non_null.describe()
        skew = float(non_null.skew()) if len(non_null) > 2 else 0.0
        q1, q3 = float(non_null.quantile(0.25)), float(non_null.quantile(0.75))
        iqr = q3 - q1
        outliers = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum()) if iqr else 0
        return {
            "kind": "numeric",
            "mean": round(float(desc["mean"]), 4),
            "median": round(float(non_null.median()), 4),
            "std": round(float(desc["std"]), 4) if len(non_null) > 1 else 0.0,
            "min": round(float(desc["min"]), 4),
            "max": round(float(desc["max"]), 4),
            "skew": round(skew, 3),
            "skew_label": "symmetric" if abs(skew) < 0.5 else ("moderate" if abs(skew) < 1 else "strong"),
            "outlier_count": outliers,
        }
    counts = non_null.astype("string").value_counts()
    return {
        "kind": "categorical",
        "n_unique": int(counts.size),
        "mode": str(counts.index[0]),
        "mode_freq_pct": round(float(counts.iloc[0] / len(non_null) * 100), 2),
        "top_values": {str(k): int(v) for k, v in counts.head(5).items()},
    }


# --------------------------------------------------------------------------- #
# Distribution data for plots
# --------------------------------------------------------------------------- #
def distribution_data(df: pd.DataFrame, column: str, bins: int = 30) -> dict[str, Any]:
    """Plot-ready distribution data for a column that has missing values.

    Numeric -> histogram bin edges/counts plus box-plot five-number summary.
    Categorical -> value-count bars.
    Also returns the *observed vs missing* split of a correlated numeric column
    so the UI can show whether the missingness shifts the distribution (MAR evidence).
    """
    s = df[column]
    non_null = s.dropna()
    payload: dict[str, Any] = {"column": column, "missing_count": int(s.isna().sum())}
    if non_null.empty:
        return {**payload, "kind": "empty"}

    if ptypes.is_numeric_dtype(non_null):
        counts, edges = np.histogram(non_null.to_numpy(dtype=float), bins=min(bins, max(5, int(np.sqrt(len(non_null))))))
        payload.update(
            kind="histogram",
            bin_edges=[round(float(e), 4) for e in edges],
            bin_labels=[f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(edges) - 1)],
            counts=[int(c) for c in counts],
            box={
                "min": float(non_null.min()),
                "q1": float(non_null.quantile(0.25)),
                "median": float(non_null.median()),
                "q3": float(non_null.quantile(0.75)),
                "max": float(non_null.max()),
            },
        )
    elif ptypes.is_datetime64_any_dtype(non_null):
        by_period = non_null.dt.to_period("M").value_counts().sort_index()
        payload.update(
            kind="timeline",
            bin_labels=[str(p) for p in by_period.index],
            counts=[int(v) for v in by_period.to_numpy()],
        )
    else:
        counts = non_null.astype("string").value_counts().head(20)
        payload.update(
            kind="bar",
            bin_labels=[str(i) for i in counts.index],
            counts=[int(v) for v in counts.to_numpy()],
        )
    payload["missingness_split"] = _missingness_split(df, column)
    return payload


def _missingness_split(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Compare other numeric columns between rows where ``column`` is present vs missing."""
    mask = df[column].isna()
    if not mask.any() or mask.all():
        return {}
    split: dict[str, Any] = {}
    for other in df.select_dtypes(include="number").columns:
        if other == column:
            continue
        present_mean, missing_mean = df.loc[~mask, other].mean(), df.loc[mask, other].mean()
        if pd.notna(present_mean) and pd.notna(missing_mean):
            split[other] = {"mean_when_present": round(float(present_mean), 4), "mean_when_missing": round(float(missing_mean), 4)}
    return dict(list(split.items())[:5])


def distribution_bundle(df: pd.DataFrame, columns: list[str] | None = None, bins: int = 30) -> dict[str, dict[str, Any]]:
    """Distribution data for every column that has at least one missing value."""
    targets = columns or [c for c in df.columns if df[c].isna().any()]
    return {c: distribution_data(df, c, bins=bins) for c in targets if c in df.columns}


# --------------------------------------------------------------------------- #
# Imputation & removal
# --------------------------------------------------------------------------- #
def impute_mean(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Fill nulls with the column mean (numeric only)."""
    return _impute_statistic(df, column, "mean")


def impute_median(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Fill nulls with the column median (numeric only, robust to skew/outliers)."""
    return _impute_statistic(df, column, "median")


def impute_mode(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Fill nulls with the most frequent value (works for any dtype)."""
    out = df.copy()
    n = int(out[column].isna().sum())
    modes = out[column].mode(dropna=True)
    if modes.empty:
        return out, f"Skipped mode imputation on '{column}': column is entirely null."
    out[column] = out[column].fillna(modes.iloc[0])
    return out, f"Imputed {n} value(s) in '{column}' with mode ({modes.iloc[0]!r})."


def _impute_statistic(df: pd.DataFrame, column: str, statistic: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if not ptypes.is_numeric_dtype(out[column]):
        raise TypeError(f"{statistic} imputation requires a numeric column; '{column}' is {out[column].dtype}.")
    n = int(out[column].isna().sum())
    value = float(getattr(out[column], statistic)())
    out[column] = out[column].fillna(value)
    return out, f"Imputed {n} value(s) in '{column}' with {statistic} ({value:.4f})."


def impute_knn(
    df: pd.DataFrame,
    column: str,
    n_neighbors: int = 5,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Multivariate KNN imputation using the other numeric columns as predictors.

    Appropriate for MAR data, where the missingness is explained by observed columns.
    """
    from sklearn.impute import KNNImputer  # local import: heavy optional dependency

    out = df.copy()
    if not ptypes.is_numeric_dtype(out[column]):
        raise TypeError(f"KNN imputation requires a numeric column; '{column}' is {out[column].dtype}.")

    numeric = out.select_dtypes(include="number")
    features = [c for c in (feature_columns or numeric.columns) if c in numeric.columns]
    if column not in features:
        features = [column, *features]
    usable = [c for c in features if numeric[c].notna().any()]
    if len(usable) < 2:
        fallback, msg = impute_median(out, column)
        return fallback, f"KNN needs >=2 usable numeric columns; fell back to median. {msg}"

    n = int(out[column].isna().sum())
    k = max(1, min(n_neighbors, int(numeric[usable].dropna().shape[0]) or 1))
    block = numeric[usable].to_numpy(dtype=float)
    mean = np.nanmean(block, axis=0)
    std = np.nanstd(block, axis=0)
    std[(std == 0) | np.isnan(std)] = 1.0
    scaled = (block - mean) / std
    filled = KNNImputer(n_neighbors=k, weights="distance").fit_transform(scaled) * std + mean
    out[column] = pd.Series(filled[:, usable.index(column)], index=out.index)
    return out, f"Imputed {n} value(s) in '{column}' with KNN (k={k}, features={usable})."


def drop_rows_with_na(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Drop rows where ``column`` is null (listwise deletion)."""
    out = df.dropna(subset=[column]).reset_index(drop=True)
    return out, f"Dropped {len(df) - len(out)} row(s) with null '{column}'."


def drop_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Remove a column that is too incomplete to be salvaged."""
    if column not in df.columns:
        return df.copy(), f"Column '{column}' not found; nothing dropped."
    pct = round(float(df[column].isna().mean() * 100), 2)
    return df.drop(columns=[column]), f"Dropped column '{column}' ({pct}% missing)."


def fill_constant(df: pd.DataFrame, column: str, value: Any) -> tuple[pd.DataFrame, str]:
    """Fill nulls with an explicit constant (e.g. 'Unknown' for MNAR categoricals)."""
    out = df.copy()
    n = int(out[column].isna().sum())
    out[column] = out[column].fillna(value)
    return out, f"Filled {n} null(s) in '{column}' with constant {value!r}."


def keep_as_is(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """No-op action, so 'do nothing' is an explicit, auditable decision."""
    return df.copy(), f"Left '{column}' unchanged."


ACTION_REGISTRY: dict[str, Callable[..., tuple[pd.DataFrame, str]]] = {
    "drop_duplicates": drop_duplicate_rows,
    "strip_whitespace": strip_whitespace,
    "unify_case": unify_case,
    "convert_numeric": convert_to_numeric,
    "clean_dates": clean_dates,
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_mode": impute_mode,
    "impute_knn": impute_knn,
    "drop_rows": drop_rows_with_na,
    "drop_column": drop_column,
    "fill_constant": fill_constant,
    "keep_as_is": keep_as_is,
}
