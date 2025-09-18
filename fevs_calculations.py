# fevs_calculations.py
import pandas as pd
import numpy as np
import re

# ---------- Pure calculations (no Streamlit) ----------
def to_pct(num, denom):
    if denom == 0 or pd.isna(num) or pd.isna(denom):
        return 0.0
    return 100.0 * float(num) / float(denom)

def likert_split(series: pd.Series):
    s = series.dropna()
    total = len(s)
    pos = (s >= 4).sum()
    neu = (s == 3).sum()
    neg = (s <= 2).sum()
    return to_pct(pos, total), to_pct(neu, total), to_pct(neg, total), total

def parse_question_list(qs_cell):
    if pd.isna(qs_cell):
        return []
    text = str(qs_cell).replace("Q.", "Q")
    parts = [p.strip() for p in text.split(",")]
    out = []
    for p in parts:
        m = re.match(r'Q?\s*(\d+)', p)
        if m:
            out.append(f"Q{int(m.group(1))}")
    return out

def prepare_population_long(pop_df: pd.DataFrame) -> pd.DataFrame:
    pop = pop_df.copy()
    pop.columns = [str(c).strip() for c in pop.columns]
    id_cols = [c for c in ["Data Set", "Random.Seed"] if c in pop.columns]
    if not id_cols:
        pop["Data_Set"] = "Set"
        id_cols = ["Data_Set"]
    # year columns like 2023, 2024, 2025
    value_cols = [c for c in pop.columns if re.search(r"\b\d{4}\b", str(c))] or \
                 [c for c in pop.columns if str(c).isdigit()]
    long = pop.melt(id_vars=id_cols, value_vars=value_cols,
                    var_name="FY", value_name="admin")
    long["FY"] = long["FY"].astype(str).str.extract(r"(\d{4})").astype(int)
    long = long.groupby("FY", as_index=False)["admin"].sum()
    return long

def compute_positive_for_questions(df: pd.DataFrame, questions):
    vals = []
    for q in questions:
        if q in df.columns:
            p, n, g, tot = likert_split(df[q])
            if tot > 0:
                vals.append(p)
    return (sum(vals)/len(vals)) if vals else None

def compute_subindex_value(idxmap: pd.DataFrame, data: pd.DataFrame,
                           subindex_name: str, fallback: float) -> float:
    if idxmap is None or idxmap.empty:
        return fallback
    needed_cols = ["Sub-Index", "Questions"]
    if not all(c in idxmap.columns for c in needed_cols):
        return fallback

    # case-insensitive exact match on Sub-Index
    mask = idxmap["Sub-Index"].astype(str).str.strip().str.lower() == subindex_name.strip().lower()
    subset = idxmap[mask]
    # if no exact match, try contains
    if subset.empty:
        mask = idxmap["Sub-Index"].astype(str).str.lower().str.contains(subindex_name.strip().lower())
        subset = idxmap[mask]
    if subset.empty:
        return fallback

    qs_all = []
    for _, row in subset.iterrows():
        qs_all.extend(parse_question_list(row["Questions"]))
    # dedupe, preserve order
    seen = set()
    qs_all = [x for x in qs_all if not (x in seen or seen.add(x))]

    val = compute_positive_for_questions(data, qs_all)
    return val if val is not None else fallback

def compute_index_value(idxmap: pd.DataFrame, data: pd.DataFrame,
                        dimension_name: str, fallback: float) -> float:
    if idxmap is None or idxmap.empty:
        return fallback

    # Find dimension column flexibly
    dim_col = None
    for c in idxmap.columns:
        if str(c).strip().lower().startswith("index-performance dimension") \
           or str(c).strip().lower() in {"dimension"}:
            dim_col = c
            break
    dim_col = dim_col or ("Index-Performance Dimension" if "Index-Performance Dimension" in idxmap.columns else None)
    if dim_col is None:
        return fallback

    # Filter rows by dimension (exact, then contains)
    mask = idxmap[dim_col].astype(str).str.strip().str.lower() == dimension_name.strip().lower()
    subset = idxmap[mask]
    if subset.empty:
        mask = idxmap[dim_col].astype(str).str.lower().str.contains(dimension_name.strip().lower())
        subset = idxmap[mask]
    if subset.empty:
        return fallback

    # If Questions exist on these rows, aggregate directly
    if "Questions" in subset.columns:
        qs_all = []
        for _, row in subset.iterrows():
            qs_all.extend(parse_question_list(row.get("Questions", "")))
        # dedupe
        qs_all = [q for i, q in enumerate(qs_all) if q not in qs_all[:i]]
        val = compute_positive_for_questions(data, qs_all)
        if val is not None:
            return val

    # Fallback: average via sub-index rows under this dimension
    if "Sub-Index" in subset.columns:
        sub_vals = []
        for _, row in subset.iterrows():
            sname = str(row.get("Sub-Index", "")).strip()
            if sname:
                v = compute_subindex_value(idxmap, data, sname, None)
                if v is not None:
                    sub_vals.append(v)
        if sub_vals:
            return sum(sub_vals)/len(sub_vals)

    return fallback
