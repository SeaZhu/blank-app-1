"""Performance Dimensions page."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from fevs_calculations import likert_split
from fevs_io import load_excel

st.set_page_config(
    page_title="Performance Dimensions · FEVS-style Dashboard",
    layout="wide",
)

st.title("Performance Dimensions")


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


def _prepare_question_metadata(
    map_df: pd.DataFrame, def_df: pd.DataFrame
) -> pd.DataFrame:
    """Return tidy metadata with question text and hierarchy."""
    if map_df is None or map_df.empty:
        return pd.DataFrame()

    # The sheet contains a title section; row with "Item.ID" is the header row.
    header_row = map_df.iloc[7].tolist() if len(map_df) > 8 else map_df.columns.tolist()
    tidy = map_df.iloc[8:].copy()
    tidy.columns = header_row
    tidy = tidy.rename(columns={c: str(c).strip() for c in tidy.columns})

    rename_map = {
        "Item.ID": "QuestionID",
        "Item.Text": "QuestionText",
        "Index": "Index",
        "Sub.Index": "SubIndex",
    }
    tidy = tidy.rename(columns=rename_map)

    cols_to_keep = [c for c in ["QuestionID", "QuestionText", "Index", "SubIndex"] if c in tidy.columns]
    tidy = tidy[cols_to_keep].copy()
    tidy = tidy.dropna(subset=["QuestionID"])

    tidy["QuestionID"] = tidy["QuestionID"].astype(str).str.strip().str.upper()
    tidy["QuestionText"] = tidy["QuestionText"].astype(str).str.strip()
    tidy["SubIndex"] = tidy["SubIndex"].astype(str).str.strip()
    tidy["Index"] = tidy["Index"].astype(str).str.strip()
    tidy["QuestionOrder"] = (
        tidy["QuestionID"].str.extract(r"(\d+)").astype(float).fillna(0).astype(int)
    )

    if def_df is not None and not def_df.empty:
        def_clean = def_df.rename(
            columns={
                "Index-Performance Dimension": "Performance Dimension",
                "Sub-Index": "SubIndex",
            }
        ).copy()
        def_clean["SubIndex"] = def_clean["SubIndex"].astype(str).str.strip()
        def_clean["Performance Dimension"] = (
            def_clean["Performance Dimension"].astype(str).str.strip()
        )
        tidy = tidy.merge(
            def_clean[["SubIndex", "Performance Dimension"]],
            on="SubIndex",
            how="left",
        )
    else:
        tidy["Performance Dimension"] = tidy["Index"]

    tidy["Performance Dimension"] = tidy["Performance Dimension"].fillna(tidy["Index"])
    tidy = tidy.drop_duplicates(subset=["QuestionID"])
    tidy = tidy.sort_values(["Performance Dimension", "SubIndex", "QuestionOrder"])
    return tidy.reset_index(drop=True)


def _compute_question_scores(raw: pd.DataFrame, question_ids: Iterable[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["FY", "QuestionID", "Positive", "Neutral", "Negative", "Responses"])

    available_questions = [q for q in question_ids if q in raw.columns]
    grouped = raw.dropna(subset=["FY"]).copy()
    grouped["FY"] = grouped["FY"].astype(int)

    for year, year_df in grouped.groupby("FY"):
        for q in available_questions:
            p, neu, neg, total = likert_split(year_df[q])
            if total == 0:
                continue
            records.append(
                {
                    "FY": int(year),
                    "QuestionID": q,
                    "Positive": p,
                    "Neutral": neu,
                    "Negative": neg,
                    "Responses": int(total),
                }
            )
    return pd.DataFrame.from_records(records)


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

metadata = _prepare_question_metadata(map_sheet, def_map)
metadata = metadata[metadata["QuestionID"].isin(raw.columns)]

if metadata.empty:
    st.error("Could not derive question metadata from the workbook.")
    st.stop()

scores = _compute_question_scores(raw, metadata["QuestionID"].unique())
if scores.empty:
    st.warning("No response data available for the selected workbook.")
    st.stop()

available_years = sorted(scores["FY"].unique())
perception_choices = {
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
}

default_years = available_years[-2:] if len(available_years) >= 2 else available_years
selected_years = st.sidebar.multiselect(
    "Survey Years",
    options=available_years,
    default=default_years,
)
if not selected_years:
    selected_years = available_years
selected_years = sorted(selected_years)

perception_label = st.sidebar.selectbox("Perception", options=list(perception_choices.keys()))
perception_column = perception_choices[perception_label]

st.caption(
    "Results show the {0} share of responses for each performance dimension, "
    "sub-index, and survey item.".format(perception_label.lower())
)

scores = scores[scores["FY"].isin(selected_years)]
if scores.empty:
    st.info("No responses for the selected filters.")
    st.stop()


def _mean_metric(subset: pd.DataFrame, year: int) -> float | None:
    series = subset.loc[subset["FY"] == year, perception_column]
    if series.empty:
        return None
    value = series.mean()
    return float(value) if pd.notna(value) else None


for dimension, dim_meta in metadata.groupby("Performance Dimension"):
    st.markdown(f"## {dimension}")
    dim_questions = dim_meta["QuestionID"].unique()
    dim_scores = scores[scores["QuestionID"].isin(dim_questions)]

    # Dimension-level metrics
    metric_cols = st.columns(len(selected_years))
    for col, year in zip(metric_cols, selected_years):
        value = _mean_metric(dim_scores, year)
        display = f"{value:.0f}%" if value is not None else "—"
        with col:
            st.metric(label=str(year), value=display)

    # Sub-index summary table
    summary_rows: list[dict[str, object]] = []
    subindex_snapshot: dict[str, dict[str, object]] = {}
    for sub_index, sub_meta in dim_meta.groupby("SubIndex"):
        sub_questions = sub_meta["QuestionID"].unique()
        sub_scores = dim_scores[dim_scores["QuestionID"].isin(sub_questions)]
        row: dict[str, object] = {
            "Sub-Index": sub_index,
            "Questions": len(sub_questions),
        }
        for year in selected_years:
            value = _mean_metric(sub_scores, year)
            row[str(year)] = value
        summary_rows.append(row)
        subindex_snapshot[sub_index] = row

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        display_columns = ["Sub-Index"] + [str(y) for y in selected_years] + ["Questions"]
        summary_df = summary_df[display_columns]
        summary_df = summary_df.sort_values("Sub-Index")
        column_config = {
            str(y): st.column_config.NumberColumn(label=str(y), format="%.0f%%")
            for y in selected_years
        }
        column_config["Questions"] = st.column_config.NumberColumn(format="%d")
        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
            column_config=column_config,
        )
    else:
        st.info("No sub-index data found for this dimension.")

    # Detailed expanders per sub-index
    for sub_index, sub_meta in dim_meta.groupby("SubIndex"):
        expander_title = f"{dimension}: {sub_index}"
        with st.expander(expander_title, expanded=False):
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
                question_scores = scores[scores["QuestionID"] == qid]
                entry: dict[str, object] = {
                    "Question": f"{qid}. {qtext}",
                }
                for year in selected_years:
                    value = _mean_metric(question_scores, year)
                    entry[str(year)] = value
                entry["Responses"] = (
                    question_scores["Responses"].sum()
                    if not question_scores.empty
                    else None
                )
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
                st.info("No questions available for this sub-index.")

    st.markdown("---")
