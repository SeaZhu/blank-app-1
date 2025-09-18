# pages/1_Overview.py
import streamlit as st
import pandas as pd
from pathlib import Path

from fevs_io import load_excel
from fevs_charts import gauge, response_rate_line
from fevs_calculations import (
    to_pct, likert_split, prepare_population_long,
    compute_subindex_value, compute_index_value
)

st.set_page_config(page_title="Overview · FEVS-style Dashboard", layout="wide")
st.title("Overview")

# --------- Cached loaders/wrappers (UI layer) ---------
@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str):
    return load_excel(fp)

@st.cache_data(show_spinner=False)
def _prepare_population_long_cached(pop_df: pd.DataFrame):
    return prepare_population_long(pop_df)

# --------- Data loading (sidebar) ---------
st.sidebar.header("Data")
default_path = Path("data/fevs_sample_data_3FYs_DataSet_5.xlsx")

if default_path.exists():
    sheets = _load_excel_cached(str(default_path))
else:
    example = Path(__file__).with_name("fevs_sample_data_3FYs_DataSet_5.xlsx")
    if example.exists():
        sheets = _load_excel_cached(str(example))
    else:
        uploaded = st.sidebar.file_uploader("Upload the Excel file", type=["xlsx"])
        if uploaded:
            sheets = load_excel(uploaded)   # un-cached stream
        else:
            st.error("Upload the Excel file or place it under ./data/")
            st.stop()

with st.sidebar.expander("Sheets Loaded", expanded=False):
    for k in sheets.keys():
        st.write("•", k)

raw = sheets.get("fevs_sample_data_3FYs_Set5")
pop = sheets.get("Population")
idxmap = sheets.get("Index-Def-Map")

if raw is None or raw.empty:
    st.error("Sheet 'fevs_sample_data_3FYs_Set5' missing.")
    st.stop()

qcols = [c for c in raw.columns if str(c).startswith("Q")]
years = sorted(raw["FY"].dropna().unique())

# --------- Filters ---------
c1, c2 = st.columns([1, 3])
with c1:
    fy = st.selectbox("Survey Year", ["All"] + [int(y) for y in years])
with c2:
    st.markdown("### Selected Survey Year")
    st.markdown(f"# {fy if fy != 'All' else 'All Years'}")

data = raw[raw["FY"] == int(fy)].copy() if fy != "All" else raw.copy()

# --------- Left panel ---------
left, right = st.columns([1.2, 2.3])

with left:
    st.write("#### Field Period")
    st.subheader("May - Jul")
    st.write("#### Sample or Census")
    st.subheader("Census")

    completed = data["Response.ID"].nunique()

    if pop is not None and not pop.empty:
        pop_long = _prepare_population_long_cached(pop)
        if fy != "All":
            admin_row = pop_long.loc[pop_long["FY"] == int(fy)]
            administered = int(admin_row["admin"].iloc[0]) if not admin_row.empty else completed
        else:
            administered = int(pop_long["admin"].sum())
    else:
        administered = completed

    st.write("#### Number of Surveys Completed")
    st.subheader(f"{completed:,}")
    st.write("#### Number of Surveys Administered")
    st.subheader(f"{int(administered):,}")
    st.write("#### Response Rate")
    st.subheader(f"{to_pct(completed, administered):.0f}%")

    st.write("#### Response Rate Over Time")
    if pop is not None and not pop.empty:
        comp_by_year = raw.groupby("FY")["Response.ID"].nunique().reset_index(name="completed")
        pop_long = _prepare_population_long_cached(pop)
        merged = comp_by_year.merge(pop_long, on="FY", how="left")
        merged["rate"] = 100 * merged["completed"] / merged["admin"]
        st.plotly_chart(response_rate_line(merged), use_container_width=True)
    else:
        st.info("Population sheet not found; cannot plot response rate over time.")

# --------- Right panel ---------
with right:
    # overall engagement baseline (avg Positive% across all items)
    pos_rates = []
    for q in qcols:
        p, n, g, tot = likert_split(data[q])
        if tot > 0:
            pos_rates.append(p)
    overall_engagement = (sum(pos_rates) / len(pos_rates)) if pos_rates else 0.0

    st.subheader("Employee Engagement Index")
    target_subs = ["Intrinsic Work Experience", "Leaders Lead", "Supervisors"]
    subcols = st.columns(3)
    sub_values = []
    for col, name in zip(subcols, target_subs):
        val = compute_subindex_value(idxmap, data, name, overall_engagement)
        sub_values.append((name, val))
        with col:
            st.plotly_chart(gauge(name, val), use_container_width=True)

    st.markdown("---")

    # Three indices from "Index-Performance Dimension"
    perf_conf = compute_index_value(idxmap, data, "Performance Confidence", overall_engagement)
    glob_sat  = compute_index_value(idxmap, data, "Global Satisfaction",  overall_engagement)
    emp_eng   = compute_index_value(idxmap, data, "Employee Engagement",  overall_engagement)

    st.metric("Performance Confidence Index", f"{perf_conf:.0f}%")
    st.markdown("---")
    st.metric("Global Satisfaction Index", f"{glob_sat:.0f}%")
    st.markdown("---")
    st.metric("Employee Engagement Index", f"{emp_eng:.0f}%")
    st.markdown("---")
