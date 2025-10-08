"""Areas of Concern page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata


st.set_page_config(
    page_title="Areas of Concern · FEVS-style Dashboard",
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
year_labels = [str(year) for year in years_to_show]

if not years_to_show:
    st.info("No fiscal year information available in the workbook.")
    st.stop()

st.sidebar.subheader("Filters")
selected_index = st.sidebar.selectbox("Index", options=available_indices, index=0)

st.title("Areas of Concern")
st.caption(
    "Each index highlights the survey item with the lowest three-year average positive "
    "response rate. Use the filter to explore the lowest-performing question for any "
    "index and inspect how perceptions have shifted over time."
)


def _format_question_text(question_id: object, question_text: object) -> str:
    """Return a readable survey item label with graceful fallbacks."""

    question_id_str = "" if pd.isna(question_id) else str(question_id).strip()
    if isinstance(question_text, str):
        question_text_str = question_text.strip()
    else:
        question_text_str = ""

    if not question_text_str:
        question_text_str = "Question text unavailable"

    if question_id_str:
        return f"{question_id_str}. {question_text_str}"
    return question_text_str


def _area_of_concern_for_index(index_name: str) -> dict[str, object] | None:
    subset = scores[(scores["Performance Dimension"] == index_name) & (scores["FY"].isin(years_to_show))]
    if subset.empty:
        return None

    best_candidate: dict[str, object] | None = None
    for qid, q_group in subset.groupby("QuestionID"):
        avg_positive = _weighted_percent(q_group, "Positive")
        if avg_positive is None:
            continue

        if best_candidate is None or avg_positive < best_candidate["avg_positive"]:
            question_text_raw = q_group["QuestionText"].iloc[0]
            subindex = q_group["SubIndexDisplay"].iloc[0]
            yearly_rows: list[dict[str, object]] = []
            for year in years_to_show:
                row = q_group[q_group["FY"] == year]
                if row.empty:
                    continue
                row_values = row.iloc[0]
                entry = {"FY": str(year)}
                for perception in ("Positive", "Neutral", "Negative"):
                    if perception in row_values.index:
                        entry[perception] = round(float(row_values[perception]), 2)
                yearly_rows.append(entry)

            if not yearly_rows:
                continue

            best_candidate = {
                "question_id": qid,
                "question_text": _format_question_text(qid, question_text_raw),
                "subindex": subindex,
                "avg_positive": avg_positive,
                "yearly_rows": yearly_rows,
                "index": index_name,
            }

    return best_candidate


def _render_card(container: st.delta_generator.DeltaGenerator, card: dict[str, object]) -> None:
    index_name = card["index"]
    question_text = card["question_text"]
    subindex = card["subindex"]
    avg_positive = card["avg_positive"]
    yearly_rows = card["yearly_rows"]

    container.markdown(f"### {index_name}")
    detail_lines = [f"**Area of Concern:** {question_text}"]
    if subindex:
        detail_lines.append(f"<span style='color: #6c757d;'>Sub-Index: {subindex}</span>")
    container.markdown("<br/>".join(detail_lines), unsafe_allow_html=True)
    if pd.isna(avg_positive):
        container.caption("Three-year average positive: N/A")
    else:
        container.caption(f"Three-year average positive: {avg_positive:.2f}%")

    chart_df = pd.DataFrame(yearly_rows)
    if chart_df.empty:
        container.info("No perception detail available for this item.")
        return

    perception_order = ["Positive", "Neutral", "Negative"]
    chart_df = chart_df.melt(id_vars="FY", value_vars=perception_order, var_name="Perception", value_name="Percent")
    chart_df = chart_df.dropna(subset=["Percent"])
    if chart_df.empty:
        container.info("No perception detail available for this item.")
        return

    chart_df["Percent"] = chart_df["Percent"].astype(float).round(2)
    chart_df["FY"] = pd.Categorical(chart_df["FY"], categories=year_labels, ordered=True)
    perception_order = ["Positive", "Neutral", "Negative"]
    chart_df["Perception"] = pd.Categorical(chart_df["Perception"], categories=perception_order, ordered=True)

    color_map = {
        "Positive": "#0B5ED7",
        "Neutral": "#6C757D",
        "Negative": "#E5533D",
    }

    fig = px.bar(
        chart_df,
        x="FY",
        y="Percent",
        color="Perception",
        color_discrete_map=color_map,
        barmode="stack",
        category_orders={"FY": year_labels, "Perception": perception_order},
        labels={"FY": "Fiscal Year", "Percent": "Percent of Responses"},
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title="Perception",
        yaxis=dict(range=[0, 100]),
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textfont_size=14, textposition="inside")

    container.plotly_chart(fig, width="stretch", config={"displaylogo": False})


selected_card = _area_of_concern_for_index(selected_index)
if selected_card is None:
    st.info("Could not determine the area of concern for the selected index.")
else:
    card_container = st.container()
    _render_card(card_container, selected_card)

