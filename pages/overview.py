# pages/1_Overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from fevs_io import load_excel
from fevs_charts import response_rate_line
from fevs_calculations import (
    to_pct,
    likert_split,
    prepare_population_long,
    compute_index_value,
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

# --------- Left panel ---------
left, right = st.columns([1.2, 2.3])

with left:
    st.write("#### Field Period")
    st.subheader("May - Jul")

    completed_counts = (
        raw.dropna(subset=["FY"])
        .assign(FY=lambda df: df["FY"].astype(int))
        .groupby("FY")["Response.ID"]
        .nunique()
        .reset_index(name="completed")
    )

    available_years = sorted(completed_counts["FY"].unique())
    trend_years = available_years[-3:] if len(available_years) >= 3 else available_years

    avg_completed = (
        completed_counts.loc[completed_counts["FY"].isin(trend_years), "completed"].mean()
        if not completed_counts.empty
        else raw["Response.ID"].nunique()
    )

    if pop is not None and not pop.empty:
        pop_long = _prepare_population_long_cached(pop)
        pop_long = pop_long.assign(FY=lambda df: df["FY"].astype(int))
        avg_administered = (
            pop_long.loc[pop_long["FY"].isin(trend_years), "admin"].mean()
            if not pop_long.empty
            else avg_completed
        )
    else:
        avg_administered = avg_completed

    avg_completed_display = int(round(avg_completed)) if pd.notna(avg_completed) else 0
    avg_admin_display = int(round(avg_administered)) if pd.notna(avg_administered) else 0
    avg_response_rate = to_pct(avg_completed_display, avg_admin_display)

    st.write("#### Avg Surveys Completed (3 FY)")
    st.subheader(f"{avg_completed_display:,}")
    st.write("#### Avg Surveys Administered (3 FY)")
    st.subheader(f"{avg_admin_display:,}")
    st.write("#### Avg Response Rate (3 FY)")
    st.subheader(f"{avg_response_rate:.0f}%")

    st.write("#### Response Rate Over Time")
    if pop is not None and not pop.empty:
        comp_by_year = completed_counts.copy()
        pop_long = _prepare_population_long_cached(pop)
        merged = comp_by_year.merge(pop_long, on="FY", how="left")
        merged = merged.dropna(subset=["admin"])
        if not merged.empty:
            merged["rate"] = 100 * merged["completed"] / merged["admin"]
            st.plotly_chart(response_rate_line(merged), use_container_width=True)
        else:
            st.info("Population sheet not found for the available fiscal years.")
    else:
        st.info("Population sheet not found; cannot plot response rate over time.")

# --------- Right panel ---------
with right:
    st.subheader("Index Positive Rates (3 FY)")

    if idxmap is None or idxmap.empty:
        st.info("Index definition sheet not found; cannot compute index trends.")
    else:
        target_indices = [
            "Employee Engagement",
            "Employee Experience",
            "Employee-Focused",
            "Foundation",
            "Global Satisfaction",
            "Goal-Oriented",
            "Performance Confidence",
        ]

        trend_records = []
        for year in trend_years:
            year_data = raw.loc[raw["FY"] == year]
            if year_data.empty:
                continue

            year_pos_rates = []
            for q in qcols:
                if q in year_data:
                    p, _, _, tot = likert_split(year_data[q])
                    if tot > 0:
                        year_pos_rates.append(p)
            fallback = sum(year_pos_rates) / len(year_pos_rates) if year_pos_rates else 0.0

            for index_name in target_indices:
                val = compute_index_value(idxmap, year_data, index_name, fallback)
                if val is None:
                    continue
                trend_records.append(
                    {
                        "Index": index_name,
                        "FY": int(year),
                        "Positive": float(val),
                    }
                )

        if trend_records:
            trend_df = pd.DataFrame(trend_records)
            trend_df = trend_df.sort_values(["Index", "FY"])
            cols = st.columns(3)
            for i, index_name in enumerate(target_indices):
                subset = trend_df[trend_df["Index"] == index_name]
                if subset.empty:
                    continue
                subset = subset.sort_values("FY").assign(
                    FY=lambda df: df["FY"].astype(str)
                )
                fig = px.line(
                    subset,
                    x="FY",
                    y="Positive",
                    markers=True,
                    title=index_name,
                )
                fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title="Positive (%)",
                    xaxis_title=None,
                )
                fig.update_xaxes(type="category")
                fig.update_yaxes(range=[0, 100])
                fig.update_traces(mode="lines+markers")
                col = cols[i % 3]
                with col:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No index results available for the selected dataset.")
