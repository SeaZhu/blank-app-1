"""Index Results page."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

PLOTLY_CONFIG = {"displaylogo": False}

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata
from fevs_style import apply_global_styles


st.set_page_config(
    page_title="Index Results · FEVS-style Dashboard",
    layout="wide",
)
apply_global_styles()

title_placeholder = st.empty()


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
metadata["Performance Dimension"] = (
    metadata["Performance Dimension"].astype(str).str.strip()
)
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
scores["QuestionOrder"] = scores["QuestionOrder"].fillna(0).astype(int)

available_indices = (
    metadata["Performance Dimension"]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda s: s.str.lower() != "other"]
    .drop_duplicates()
    .sort_values(key=lambda s: s.str.lower())
    .tolist()
)
if not available_indices:
    st.error("No index definitions available.")
    st.stop()

available_years = sorted(scores["FY"].unique())
years_to_show = available_years[-3:] if len(available_years) >= 3 else available_years

st.sidebar.subheader("Filters")
selected_index = st.sidebar.selectbox(
    "Index",
    options=available_indices,
    index=0,
)
perception_options = ["Positive", "Negative", "All"]
perception_choice = st.sidebar.selectbox(
    "Perception",
    options=perception_options,
    index=0,
    help="Show question detail for the selected response perception.",
)

title_placeholder.title(f"Index Results: {selected_index}")

selected_scores = scores[
    (scores["Performance Dimension"] == selected_index)
    & (scores["FY"].isin(years_to_show))
].copy()

if selected_scores.empty:
    st.info("No responses available for the selected filters.")
    st.stop()

selected_metadata = metadata[metadata["Performance Dimension"] == selected_index].copy()
selected_metadata["SubIndex"] = selected_metadata["SubIndex"].fillna("").astype(str).str.strip()

ordered_subindices: list[str] = []
for value in selected_metadata["SubIndex"]:
    label = value if value else "Ungrouped Items"
    if label not in ordered_subindices:
        ordered_subindices.append(label)

if not ordered_subindices and not selected_scores.empty:
    ordered_subindices = ["Ungrouped Items"]

trend_rows: list[dict[str, object]] = []
index_label = f"{selected_index} (Index)"
for year in years_to_show:
    year_df = selected_scores[selected_scores["FY"] == year]
    pos_val = _weighted_percent(year_df, "Positive")
    if pos_val is not None:
        trend_rows.append(
            {
                "FY": year,
                "Label": index_label,
                "SeriesLabel": f"{index_label} (Positive)",
                "Perception": "Positive",
                "Percent": pos_val,
            }
        )
    neg_val = _weighted_percent(year_df, "Negative")
    if neg_val is not None:
        trend_rows.append(
            {
                "FY": year,
                "Label": index_label,
                "SeriesLabel": f"{index_label} (Negative)",
                "Perception": "Negative",
                "Percent": neg_val,
            }
        )

for subindex in ordered_subindices:
    subset = selected_scores[selected_scores["SubIndexDisplay"] == subindex]
    if subset.empty:
        continue
    for year in years_to_show:
        year_df = subset[subset["FY"] == year]
        pos_val = _weighted_percent(year_df, "Positive")
        if pos_val is not None:
            trend_rows.append(
                {
                    "FY": year,
                    "Label": subindex,
                    "SeriesLabel": f"{subindex} (Positive)",
                    "Perception": "Positive",
                    "Percent": pos_val,
                }
            )
        neg_val = _weighted_percent(year_df, "Negative")
        if neg_val is not None:
            trend_rows.append(
                {
                    "FY": year,
                    "Label": subindex,
                    "SeriesLabel": f"{subindex} (Negative)",
                    "Perception": "Negative",
                    "Percent": neg_val,
                }
            )

trend_df = pd.DataFrame(trend_rows)
if not trend_df.empty:
    trend_df["Percent"] = trend_df["Percent"].round(2)

if perception_choice == "All":
    chart_perceptions = ("Positive", "Negative")
    chart_title = "Positive & Negative Responses"
else:
    chart_perceptions = (perception_choice,)
    chart_title = f"{perception_choice} Responses"

chart_df = trend_df[trend_df["Perception"].isin(chart_perceptions)].copy()

index_chart_df = chart_df[chart_df["Label"] == index_label].copy()
subindex_chart_df = chart_df[chart_df["Label"] != index_label].copy()

index_col, subindex_col = st.columns(2)

with index_col:
    if index_chart_df.empty:
        st.info("No perception data available for the selected index.")
    else:
        index_chart_df = index_chart_df.sort_values(["SeriesLabel", "FY"])
        index_chart_df["FY"] = index_chart_df["FY"].astype(str)
        index_fig = px.line(
            index_chart_df,
            x="FY",
            y="Percent",
            color="SeriesLabel",
            markers=True,
            title=f"{chart_title} · Index",
        )
        index_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_title="Percent",
            xaxis_title=None,
            legend_title="Series",
        )
        index_fig.update_yaxes(range=[0, 100])
        index_fig.update_xaxes(type="category")
        st.plotly_chart(index_fig, config=PLOTLY_CONFIG)

with subindex_col:
    if subindex_chart_df.empty:
        st.info("No perception data available for the sub-indices.")
    else:
        subindex_chart_df = subindex_chart_df.sort_values(["SeriesLabel", "FY"])
        subindex_chart_df["FY"] = subindex_chart_df["FY"].astype(str)
        subindex_fig = px.line(
            subindex_chart_df,
            x="FY",
            y="Percent",
            color="SeriesLabel",
            markers=True,
            title=f"{chart_title} · Sub-Indices",
        )
        subindex_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_title="Percent",
            xaxis_title=None,
            legend_title="Series",
        )
        subindex_fig.update_yaxes(range=[0, 100])
        subindex_fig.update_xaxes(type="category")
        st.plotly_chart(subindex_fig, config=PLOTLY_CONFIG)

st.markdown("---")

perception_columns: dict[str, Iterable[str]] = {
    "All": ("Positive", "Neutral", "Negative"),
    "Positive": ("Positive",),
    "Negative": ("Negative",),
}
selected_perceptions = perception_columns[perception_choice]

for subindex in ordered_subindices:
    subset = selected_scores[selected_scores["SubIndexDisplay"] == subindex]
    if subset.empty:
        continue

    expander_title = f"{selected_index}: {subindex}"
    with st.expander(expander_title, expanded=False):
        question_rows: list[dict[str, object]] = []
        grouped = subset.sort_values("QuestionOrder").groupby("QuestionID", sort=False)
        for qid, q_group in grouped:
            question_text_raw = q_group["QuestionText"].iloc[0]
            question_text = (
                str(question_text_raw).strip()
                if pd.notna(question_text_raw) and str(question_text_raw).strip()
                else "Question text unavailable"
            )
            row: dict[str, object] = {"Question": f"{qid}. {question_text}"}
            for perception in selected_perceptions:
                for year in years_to_show:
                    value_series = q_group.loc[q_group["FY"] == year, perception]
                    value = float(value_series.iloc[0]) if not value_series.empty else None
                    column_label = f"{year} {perception}"
                    row[column_label] = value
            question_rows.append(row)

        if not question_rows:
            st.info("No questions available for this sub-index.")
            continue

        question_df = pd.DataFrame(question_rows)
        ordered_columns = ["Question"]
        for perception in selected_perceptions:
            for year in years_to_show:
                ordered_columns.append(f"{year} {perception}")
        question_df = question_df[ordered_columns]

        column_config: dict[str, st.column_config.Column] = {"Question": st.column_config.TextColumn()}
        for perception in selected_perceptions:
            for year in years_to_show:
                column_config[f"{year} {perception}"] = st.column_config.NumberColumn(
                    label=f"{year} {perception}",
                    format="%.0f%%",
                )

        st.dataframe(
            question_df,
            hide_index=True,
            use_container_width=True,
            column_config=column_config,
        )
