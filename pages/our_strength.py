"""Our Strength page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata


st.set_page_config(
    page_title="Our Strength · FEVS-style Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


def _weighted_percent(df: pd.DataFrame, column: str) -> float | None:
    """Return the response-share weighted value for the given perception column."""

    if column not in df.columns:
        return None
    valid = df.dropna(subset=[column, "Responses"])
    if valid.empty:
        return None
    total = valid["Responses"].sum()
    if total == 0:
        return None
    value = (valid[column] * valid["Responses"]).sum() / total
    return float(value)


def _trend_label(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "Insufficient data"

    diff = end - start
    if diff >= 3:
        return "Improving"
    if diff >= 1:
        return "Slight ↑"
    if diff <= -3:
        return "Declining"
    if diff <= -1:
        return "Slight ↓"
    return "Stable"


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
metadata = metadata.dropna(subset=["Performance Dimension"])
metadata["Performance Dimension"] = metadata["Performance Dimension"].astype(str).str.strip()
metadata["SubIndex"] = metadata["SubIndex"].fillna("").astype(str).str.strip()
metadata = metadata[metadata["Performance Dimension"].str.lower() != "other"]

if metadata.empty:
    st.error("Could not derive question metadata from the workbook.")
    st.stop()

question_ids = metadata["QuestionID"].unique()
scores = compute_question_scores(raw, question_ids)
if scores.empty:
    st.warning("No response data available for the selected workbook.")
    st.stop()

scores = scores.merge(
    metadata[
        [
            "QuestionID",
            "QuestionText",
            "SubIndex",
            "Performance Dimension",
            "QuestionOrder",
        ]
    ],
    on="QuestionID",
    how="left",
)

scores = scores.dropna(subset=["Performance Dimension"])
scores["FY"] = scores["FY"].astype(int)
scores["Performance Dimension"] = scores["Performance Dimension"].astype(str).str.strip()
scores["SubIndex"] = scores["SubIndex"].fillna("").astype(str).str.strip()
scores["SubIndexDisplay"] = scores["SubIndex"].apply(lambda x: x if x else "Ungrouped Items")

available_years = sorted(scores["FY"].unique())
if not available_years:
    st.info("No fiscal year information available in the workbook.")
    st.stop()

years_to_show = available_years[-3:] if len(available_years) >= 3 else available_years
year_labels = [str(year) for year in years_to_show]

st.title("Our Strength")
st.caption(
    "Top-performing survey items are ranked by their three-year average positive "
    "response rate, revealing the questions where employees consistently respond "
    "most favorably."
)

question_summaries: list[dict[str, object]] = []
for qid, q_group in scores.groupby("QuestionID"):
    subset = q_group[q_group["FY"].isin(years_to_show)]
    if subset.empty:
        continue

    avg_positive = _weighted_percent(subset, "Positive")
    if avg_positive is None:
        continue

    per_year: dict[str, float | None] = {}
    for year in years_to_show:
        year_value = _weighted_percent(subset[subset["FY"] == year], "Positive")
        per_year[str(year)] = round(year_value, 2) if year_value is not None else None

    question_summaries.append(
        {
            "QuestionID": qid,
            "QuestionText": subset["QuestionText"].iloc[0],
            "Index": subset["Performance Dimension"].iloc[0],
            "SubIndex": subset["SubIndexDisplay"].iloc[0],
            "AveragePositive": float(avg_positive),
            "PerYear": per_year,
            "TrendStart": per_year.get(str(years_to_show[0])),
            "TrendEnd": per_year.get(str(years_to_show[-1])),
        }
    )

if not question_summaries:
    st.info("Unable to calculate strength items for the available data.")
    st.stop()

question_summaries.sort(key=lambda item: item["AveragePositive"], reverse=True)
top_strengths = question_summaries[:5]

if not top_strengths:
    st.info("No survey items qualified for the strength ranking.")
    st.stop()

chart_records: list[dict[str, object]] = []
for item in top_strengths:
    question_label = f"{item['QuestionID']}. {item['QuestionText']}"
    per_year = item["PerYear"]
    for year in years_to_show:
        year_str = str(year)
        value = per_year.get(year_str)
        if value is None:
            continue
        chart_records.append(
            {
                "Survey Item": question_label,
                "FY": year_str,
                "Positive": round(float(value), 2),
            }
        )

if chart_records:
    chart_df = pd.DataFrame(chart_records)
    chart_df["FY"] = pd.Categorical(chart_df["FY"], categories=year_labels, ordered=True)

    fig = px.line(
        chart_df,
        x="FY",
        y="Positive",
        color="Survey Item",
        markers=True,
        category_orders={"FY": year_labels},
        labels={"FY": "Fiscal Year", "Positive": "% Positive"},
    )
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=20, t=60, b=10),
        legend_title="Survey Item",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
    )
    fig.update_xaxes(type="category")
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}%", line_width=3)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No chartable data found for the top strengths.")


table_rows: list[dict[str, object]] = []
for item in top_strengths:
    per_year = item["PerYear"]
    row: dict[str, object] = {
        "Survey Item": f"{item['QuestionID']}. {item['QuestionText']}",
        "Index": item["Index"],
        "Sub-Index": item["SubIndex"] if item["SubIndex"] else "Ungrouped Items",
    }
    for year in years_to_show:
        value = per_year.get(str(year))
        row[str(year)] = value
    row["3-Year Avg"] = round(item["AveragePositive"], 2)
    row["Trend"] = _trend_label(item["TrendStart"], item["TrendEnd"])
    table_rows.append(row)

if table_rows:
    table_df = pd.DataFrame(table_rows)
    ordered_columns = ["Survey Item", "Index", "Sub-Index"] + year_labels + ["3-Year Avg", "Trend"]
    table_df = table_df[ordered_columns]

    column_config: dict[str, st.column_config.Column | st.column_config.TextColumn] = {
        "Survey Item": st.column_config.TextColumn(label="Survey Item"),
        "Index": st.column_config.TextColumn(label="Index"),
        "Sub-Index": st.column_config.TextColumn(label="Sub-Index"),
        "3-Year Avg": st.column_config.NumberColumn(label="3-Year Avg", format="%.2f%%"),
    }
    for year in years_to_show:
        column_config[str(year)] = st.column_config.NumberColumn(label=str(year), format="%.1f%%")
    column_config["Trend"] = st.column_config.TextColumn(label="Trend")

    st.dataframe(
        table_df,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )
else:
    st.info("No tabular data available for the strength summary.")
