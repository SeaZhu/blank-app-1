"""Opportunities to Improve page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata


st.set_page_config(
    page_title="Opportunities to Improve · FEVS-style Dashboard",
    layout="wide",
)


title_placeholder = st.empty()


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


DEFAULT_PATH = Path("data/fevs_sample_data_3FYs_DataSet_5.xlsx")

# --------- Data loading ---------
if DEFAULT_PATH.exists():
    sheets = _load_excel_cached(str(DEFAULT_PATH))
else:
    example = Path(__file__).with_name("fevs_sample_data_3FYs_DataSet_5.xlsx")
    if example.exists():
        sheets = _load_excel_cached(str(example))
    else:
        st.error("Upload the Excel file or place it under ./data/ before launching.")
        st.stop()

raw = sheets.get("fevs_sample_data_3FYs_Set5")
map_sheet = sheets.get("Index-Qns-Map")
def_map = sheets.get("Index-Def-Map")

if raw is None or raw.empty:
    st.error("Sheet 'fevs_sample_data_3FYs_Set5' missing or empty.")
    st.stop()

metadata = prepare_question_metadata(map_sheet, def_map)
metadata = metadata[metadata["QuestionID"].isin(raw.columns)]
metadata["SubIndex"] = metadata["SubIndex"].fillna("").astype(str).str.strip()
metadata["Performance Dimension"] = (
    metadata["Performance Dimension"].fillna("").astype(str).str.strip()
)

if metadata.empty:
    st.error("Could not derive question metadata from the workbook.")
    st.stop()

question_ids = metadata["QuestionID"].unique()
scores = compute_question_scores(raw, question_ids)

if scores.empty:
    st.warning("No response data available for the selected workbook.")
    st.stop()

scores = scores.merge(
    metadata[["QuestionID", "QuestionText", "Performance Dimension", "SubIndex"]],
    on="QuestionID",
    how="left",
)

scores = scores.dropna(subset=["QuestionID", "FY"])
scores["FY"] = scores["FY"].astype(int)
scores["QuestionText"] = scores["QuestionText"].fillna("").astype(str)
scores["Performance Dimension"] = scores["Performance Dimension"].fillna("").astype(str)
scores["SubIndex"] = scores["SubIndex"].fillna("").astype(str)

available_years = sorted(scores["FY"].unique())
years_to_show = available_years[-3:] if len(available_years) >= 3 else available_years

if not years_to_show:
    st.warning("No fiscal year data available for opportunities analysis.")
    st.stop()

title_placeholder.title("Opportunities to Improve")
st.caption(
    "Lowest five survey items by share of positive responses across the most recent three fiscal years."
)

recent_scores = scores[scores["FY"].isin(years_to_show)].copy()

if recent_scores.empty:
    st.info("No survey item responses available for the selected time frame.")
    st.stop()

# Calculate response-weighted positive percentages across years.
recent_scores["PositiveWeighted"] = recent_scores["Positive"] * recent_scores["Responses"]
agg = (
    recent_scores.groupby("QuestionID", as_index=False)
    .agg(
        {
            "PositiveWeighted": "sum",
            "Responses": "sum",
            "QuestionText": "first",
            "Performance Dimension": "first",
            "SubIndex": "first",
        }
    )
    .rename(columns={"PositiveWeighted": "PositiveWeightedSum", "Responses": "ResponseSum"})
)

agg = agg[agg["ResponseSum"] > 0]
if agg.empty:
    st.info("No aggregated question responses available.")
    st.stop()

agg["ThreeYearAverage"] = agg["PositiveWeightedSum"] / agg["ResponseSum"]
agg["SubIndex"] = agg["SubIndex"].replace({"": "Ungrouped Items"})
agg["Performance Dimension"] = agg["Performance Dimension"].replace({"": "Not Classified"})

lowest_five = agg.nsmallest(5, "ThreeYearAverage").reset_index(drop=True)

if lowest_five.empty:
    st.info("All survey items have identical positive response rates.")
    st.stop()

lowest_ids = lowest_five["QuestionID"].tolist()
order_map = {qid: pos for pos, qid in enumerate(lowest_ids)}

chart_df = recent_scores[recent_scores["QuestionID"].isin(lowest_ids)].copy()
chart_df["FY"] = chart_df["FY"].astype(str)
chart_df["QuestionLabel"] = chart_df.apply(
    lambda row: f"{row['QuestionID']}. "
    + (
        row["QuestionText"].strip()
        if row["QuestionText"].strip()
        else "Question text unavailable"
    ),
    axis=1,
)
chart_df["QuestionOrder"] = chart_df["QuestionID"].map(order_map)
chart_df["Positive"] = chart_df["Positive"].round(2)

chart_df = chart_df.sort_values(["QuestionOrder", "FY"])

fig = px.line(
    chart_df,
    x="FY",
    y="Positive",
    color="QuestionLabel",
    markers=True,
    category_orders={"FY": [str(year) for year in years_to_show]},
    labels={"Positive": "Positive Responses (%)", "FY": "Fiscal Year", "QuestionLabel": "Survey Item"},
    title="Lowest Five Survey Items by Positive Response",
)
fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), legend_title="Survey Item")
fig.update_yaxes(range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

# Build summary table
yearly = (
    recent_scores[recent_scores["QuestionID"].isin(lowest_ids)]
    .pivot_table(index="QuestionID", columns="FY", values="Positive")
    .rename(columns=lambda c: str(int(c)))
)

summary = lowest_five.merge(yearly, on="QuestionID", how="left")

def _format_question_text(question_id: str, question_text: str) -> str:
    question_text = question_text.strip()
    if not question_text:
        question_text = "Question text unavailable"
    return f"{question_id}. {question_text}"


def _describe_trend(row: pd.Series, years: list[int]) -> str:
    values = []
    for year in years:
        value = row.get(str(year))
        values.append(value if pd.notna(value) else None)
    filtered = [v for v in values if v is not None]
    if len(filtered) < 2:
        return "—"
    delta = filtered[-1] - filtered[0]
    if abs(delta) < 0.5:
        return "Stable"
    if delta > 0:
        return "Slight ↑" if delta < 2 else "Improving"
    return "Slight ↓" if delta > -2 else "Declining"


year_columns = [str(year) for year in years_to_show]
summary["Survey Item"] = summary.apply(
    lambda row: _format_question_text(row["QuestionID"], row["QuestionText"]),
    axis=1,
)
summary["Index"] = summary["Performance Dimension"].replace({"": "Not Classified"})
summary["Sub-Index"] = summary["SubIndex"].replace({"": "Ungrouped Items"})

for column in year_columns:
    if column in summary:
        summary[column] = summary[column].round(1)

summary["3-Year Avg"] = summary["ThreeYearAverage"].round(1)
summary["Trend"] = summary.apply(_describe_trend, axis=1, years=years_to_show)

display_columns = ["Survey Item", "Index", "Sub-Index"] + year_columns + ["3-Year Avg", "Trend"]
summary = summary[display_columns]

column_config: dict[str, st.column_config.Column] = {
    "Survey Item": st.column_config.TextColumn(),
    "Index": st.column_config.TextColumn(),
    "Sub-Index": st.column_config.TextColumn(label="Sub Index"),
}
for column in year_columns + ["3-Year Avg"]:
    column_config[column] = st.column_config.NumberColumn(label=column, format="%.1f%%")
column_config["Trend"] = st.column_config.TextColumn()

st.markdown("\n")
st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True,
    column_config=column_config,
)

