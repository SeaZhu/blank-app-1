"""Strength Areas page."""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata

st.set_page_config(page_title="Strength Areas · FEVS-style Dashboard", layout="wide")

st.title("Strength Areas")


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


def _mean_metric(subset: pd.DataFrame, year: int, column: str) -> float | None:
    series = subset.loc[subset["FY"] == year, column]
    if series.empty:
        return None
    value = series.mean()
    return float(value) if pd.notna(value) else None


# --------- Data loading (sidebar) ---------
st.sidebar.header("Data")
DEFAULT_PATH = Path("data/fevs_sample_data_3FYs_DataSet_5.xlsx")

if DEFAULT_PATH.exists():
    sheets = _load_excel_cached(str(DEFAULT_PATH))
else:
    example = Path(__file__).with_name("fevs_sample_data_3FYs_DataSet_5.xlsx")
    if example.exists():
        sheets = _load_excel_cached(str(example))
    else:
        uploaded = st.sidebar.file_uploader("Upload the Excel file", type=["xlsx"])
        if uploaded is None:
            st.error("Upload the Excel file or place it under ./data/")
            st.stop()
        sheets = load_excel(uploaded)

raw = sheets.get("fevs_sample_data_3FYs_Set5")
map_sheet = sheets.get("Index-Qns-Map")
def_map = sheets.get("Index-Def-Map")

if raw is None or raw.empty:
    st.error("Sheet 'fevs_sample_data_3FYs_Set5' missing or empty.")
    st.stop()

metadata = prepare_question_metadata(map_sheet, def_map)
metadata = metadata[metadata["QuestionID"].isin(raw.columns)]

if metadata.empty:
    st.error("Could not derive question metadata from the workbook.")
    st.stop()

scores = compute_question_scores(raw, metadata["QuestionID"].unique())
if scores.empty:
    st.warning("No response data available for the selected workbook.")
    st.stop()

available_years = sorted(scores["FY"].unique())
default_years = available_years[-3:] if len(available_years) >= 3 else available_years
selected_years = st.sidebar.multiselect(
    "Survey Years",
    options=available_years,
    default=default_years,
)
if not selected_years:
    selected_years = available_years
selected_years = sorted(selected_years)

perception_choices = {
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
}
perception_label = st.sidebar.selectbox("Perception", options=list(perception_choices.keys()))
perception_column = perception_choices[perception_label]

scores = scores[scores["FY"].isin(selected_years)]
if scores.empty:
    st.info("No responses for the selected filters.")
    st.stop()

scores_with_meta = scores.merge(
    metadata[["QuestionID", "Index", "SubIndex", "QuestionText"]],
    on="QuestionID",
    how="left",
)
scores_with_meta = scores_with_meta.dropna(subset=["Index"])

if scores_with_meta.empty:
    st.info("No index data available after applying filters.")
    st.stop()

# Determine strongest index based on positive perception across the selected years.
index_positive = (
    scores_with_meta.groupby(["Index", "FY"])["Positive"].mean().reset_index()
)
index_rank = index_positive.groupby("Index")["Positive"].mean().reset_index(name="AveragePositive")

if index_rank.empty:
    st.info("No index-level scores found.")
    st.stop()

top_index_row = index_rank.sort_values("AveragePositive", ascending=False).iloc[0]
top_index = top_index_row["Index"]
top_index_scores = index_positive[index_positive["Index"] == top_index].sort_values("FY")

top_min = float(top_index_scores["Positive"].min())
top_max = float(top_index_scores["Positive"].max())
top_avg = float(top_index_row["AveragePositive"])

st.markdown(f"### {top_index}")

metric_cols = st.columns(len(selected_years))
for col, year in zip(metric_cols, selected_years):
    year_value = top_index_scores.loc[top_index_scores["FY"] == year, "Positive"]
    display = f"{float(year_value.iloc[0]):.0f}%" if not year_value.empty else "—"
    with col:
        st.metric(label=str(year), value=display)

summary_cols = st.columns(2)
with summary_cols[0]:
    st.metric("Average Positive", f"{top_avg:.1f}%")
threshold_label = (
    f"{len(selected_years)}-Year Positive Threshold"
    if len(selected_years) > 1
    else "Positive Threshold"
)
with summary_cols[1]:
    st.metric(threshold_label, f"{top_min:.1f}%")

chart_data = top_index_scores.rename(columns={"Positive": "Positive %"})
chart = (
    alt.Chart(chart_data)
    .mark_line(point=True)
    .encode(
        x=alt.X("FY:O", title="Fiscal Year"),
        y=alt.Y("Positive %:Q", title="Positive %", scale=alt.Scale(domain=[0, 100])),
        tooltip=["FY:O", alt.Tooltip("Positive %:Q", format=".1f")],
    )
)
st.altair_chart(chart, use_container_width=True)

year_descriptions = ", ".join(
    f"{int(row.FY)}: {row.Positive:.0f}%" for row in top_index_scores.itertuples()
)
st.caption(
    f"The {top_index} remained the strongest index because its positive perception never fell "
    f"below {top_min:.0f}% and peaked at {top_max:.0f}% across the selected years "
    f"({year_descriptions})."
)

st.markdown("#### Sub-Index trends")

metric_columns = ["Positive"]
if perception_column != "Positive":
    metric_columns.append(perception_column)

subindex_scores = (
    scores_with_meta[scores_with_meta["Index"] == top_index]
    .groupby(["SubIndex", "FY"])[metric_columns]
    .mean()
    .reset_index()
)

if subindex_scores.empty:
    st.info("No sub-index data available for the strongest index.")
else:
    sub_chart = (
        alt.Chart(subindex_scores.rename(columns={"Positive": "Positive %"}))
        .mark_line(point=True)
        .encode(
            x=alt.X("FY:O", title="Fiscal Year"),
            y=alt.Y("Positive %:Q", title="Positive %", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("SubIndex:N", title="Sub-Index"),
            tooltip=[
                "SubIndex:N",
                "FY:O",
                alt.Tooltip("Positive %:Q", format=".1f"),
            ],
        )
    )
    st.altair_chart(sub_chart, use_container_width=True)

    summary_rows: list[dict[str, object]] = []
    subindex_snapshot: dict[str, dict[str, object]] = {}
    for sub_index, subset in scores_with_meta[scores_with_meta["Index"] == top_index].groupby(
        "SubIndex"
    ):
        row: dict[str, object] = {"Sub-Index": sub_index}
        for year in selected_years:
            value = _mean_metric(subset, year, perception_column)
            row[str(year)] = value
        row["Questions"] = subset["QuestionID"].nunique()
        summary_rows.append(row)
        subindex_snapshot[sub_index] = row

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        display_columns = ["Sub-Index"] + [str(y) for y in selected_years] + ["Questions"]
        summary_df = summary_df[display_columns]
        column_config = {
            str(y): st.column_config.NumberColumn(label=str(y), format="%.0f%%")
            for y in selected_years
        }
        column_config["Questions"] = st.column_config.NumberColumn(format="%d")
        st.dataframe(
            summary_df.sort_values("Sub-Index"),
            hide_index=True,
            use_container_width=True,
            column_config=column_config,
        )

st.markdown("#### Questions contributing to the strength")

question_strength = (
    scores_with_meta[scores_with_meta["Index"] == top_index]
    .groupby(["QuestionID", "QuestionText"])["Positive"]
    .mean()
    .reset_index()
)

if not question_strength.empty:
    question_strength = question_strength.sort_values("Positive", ascending=False)
    top_highlights = [
        f"<li><strong>{row.QuestionID}.</strong> {row.QuestionText} averaged {row.Positive:.1f}% positive responses across the selected years.</li>"
        for row in question_strength.itertuples()
    ][:3]
    highlight_list = "".join(top_highlights)
    explanation_intro = (
        f"The index stays above the {top_min:.0f}% positive threshold because its questions maintain consistently strong perception scores."
    )
    st.markdown(
        f"<div style='font-size:16px; font-weight:600; color:#1f1f1f;'>{explanation_intro}</div>",
        unsafe_allow_html=True,
    )
    if highlight_list:
        st.markdown(
            f"<ul style='font-size:15px; font-weight:500; color:#1f1f1f; margin-top:0.5rem;'>{highlight_list}</ul>",
            unsafe_allow_html=True,
        )

top_questions = metadata[metadata["Index"] == top_index]
top_questions = top_questions.sort_values(["SubIndex", "QuestionOrder"])

for sub_index, sub_meta in top_questions.groupby("SubIndex"):
    with st.expander(f"{top_index}: {sub_index}", expanded=False):
        snapshot = subindex_snapshot.get(sub_index, {})
        if snapshot:
            metric_cols = st.columns(len(selected_years))
            for col, year in zip(metric_cols, selected_years):
                value = snapshot.get(str(year))
                display = f"{value:.0f}%" if isinstance(value, (int, float)) else "—"
                with col:
                    st.metric(label=str(year), value=display)

        question_rows: list[dict[str, object]] = []
        for _, q_row in sub_meta.sort_values("QuestionOrder").iterrows():
            qid = q_row["QuestionID"]
            qtext = q_row["QuestionText"]
            question_scores = scores_with_meta[
                (scores_with_meta["QuestionID"] == qid)
                & (scores_with_meta["SubIndex"] == sub_index)
            ]
            entry: dict[str, object] = {"Question": f"{qid}. {qtext}"}
            for year in selected_years:
                value = _mean_metric(question_scores, year, perception_column)
                entry[str(year)] = value
            responses = question_scores.get("Responses")
            if responses is not None and not responses.isna().all():
                response_sum = responses.sum()
                entry["Responses"] = int(response_sum) if pd.notna(response_sum) else None
            else:
                entry["Responses"] = None
            question_rows.append(entry)

        if question_rows:
            question_df = pd.DataFrame(question_rows)
            display_columns = ["Question"] + [str(y) for y in selected_years] + ["Responses"]
            question_df = question_df[display_columns]
            column_config = {
                str(y): st.column_config.NumberColumn(label=str(y), format="%.0f%%")
                for y in selected_years
            }
            column_config["Responses"] = st.column_config.NumberColumn(format="%d")
            st.dataframe(
                question_df,
                hide_index=True,
                use_container_width=True,
                column_config=column_config,
            )
        else:
            st.info("No questions found for this sub-index.")

st.caption(
    "Perception values in the tables respect the selected perception filter, while the "
    "strongest index determination is based on positive responses across all selected years."
)

