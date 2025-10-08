"""Survey Item Results page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata


st.set_page_config(
    page_title="Survey Item Results · FEVS-style Dashboard",
    layout="wide",
)


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
perception_options = ["Positive", "Negative", "Neutral"]
perception_choice = st.sidebar.selectbox(
    "Perception",
    options=perception_options,
    index=0,
    help="Show question detail for the selected response perception.",
)

perception_column = perception_choice

page_title = f"Survey Item Results: {selected_index}"
title_placeholder.title(page_title)

selected_scores = scores[
    (scores["Performance Dimension"] == selected_index)
    & (scores["FY"].isin(years_to_show))
].copy()

if selected_scores.empty:
    st.info("No responses available for the selected filters.")
    st.stop()

ordered_subindices: list[str] = []
for value in selected_scores["SubIndexDisplay"].unique():
    if value not in ordered_subindices:
        ordered_subindices.append(value)

if not ordered_subindices:
    ordered_subindices = ["Ungrouped Items"]

for position, subindex in enumerate(ordered_subindices, start=1):
    subset = selected_scores[selected_scores["SubIndexDisplay"] == subindex]
    if subset.empty:
        continue

    st.markdown(f"### {subindex}")

    summary_rows: list[dict[str, object]] = []
    summary_row: dict[str, object] = {"Question": f"{subindex} · Average"}
    for year in years_to_show:
        year_df = subset[subset["FY"] == year]
        summary_row[str(year)] = _weighted_percent(year_df, perception_column)
    summary_rows.append(summary_row)

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
        for year in years_to_show:
            value_series = q_group.loc[q_group["FY"] == year, perception_column]
            value = float(value_series.iloc[0]) if not value_series.empty else None
            row[str(year)] = value
        question_rows.append(row)

    display_df = pd.DataFrame(summary_rows + question_rows)

    ordered_columns: list[str] = ["Question"] + [str(year) for year in years_to_show]
    display_df = display_df[ordered_columns]

    column_config: dict[str, st.column_config.Column] = {
        "Question": st.column_config.TextColumn(label="Survey Item"),
    }
    for year in years_to_show:
        column_config[str(year)] = st.column_config.NumberColumn(
            label=str(year),
            format="%.0f%%",
        )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )

    if position < len(ordered_subindices):
        st.markdown("---")

