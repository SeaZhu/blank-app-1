"""Our Strength page."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata
from fevs_style import apply_global_styles


st.set_page_config(
    page_title="Our Strength · FEVS-style Dashboard",
    layout="wide",
)
apply_global_styles()


PERCEPTION_ORDER = ["Positive", "Neutral", "Negative"]
COLOR_MAP = {
    "Positive": "#0B5ED7",
    "Neutral": "#6C757D",
    "Negative": "#E5533D",
}
PLOTLY_CONFIG = {"displaylogo": False}


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


def _render_sidebar_legend() -> None:
    st.sidebar.markdown("**Legend**")
    for label in PERCEPTION_ORDER:
        color = COLOR_MAP[label]
        st.sidebar.markdown(
            (
                "<div style='display:flex; align-items:center; gap:0.5rem;'>"
                f"<span style='width:0.85rem; height:0.85rem; background:{color}; display:inline-block;"
                " border-radius:0.2rem;'></span>"
                f"<span>{label}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )



def _perception_chart(
    scores_df: pd.DataFrame,
    question_id: object,
    *,
    years: Iterable[int],
    title: str,
    text_size: int = 13,
    height: int = 420,
) -> go.Figure | None:
    year_list = list(years)
    subset = scores_df[(scores_df["QuestionID"] == question_id) & (scores_df["FY"].isin(year_list))]
    if subset.empty:
        return None

    subset = subset.copy()
    subset["FY"] = subset["FY"].astype(int).astype(str)
    available_columns = [col for col in PERCEPTION_ORDER if col in subset.columns]
    if not available_columns:
        return None

    perception_df = subset[["FY"] + available_columns].drop_duplicates(subset=["FY"])
    melted = perception_df.melt(
        id_vars="FY",
        value_vars=available_columns,
        var_name="Perception",
        value_name="Percent",
    ).dropna(subset=["Percent"])

    if melted.empty:
        return None

    year_labels = [str(year) for year in year_list]
    melted["FY"] = pd.Categorical(melted["FY"], categories=year_labels, ordered=True)
    melted["Percent"] = melted["Percent"].astype(float).round(2)
    melted["Perception"] = pd.Categorical(melted["Perception"], categories=PERCEPTION_ORDER, ordered=True)

    fig = px.bar(
        melted,
        x="FY",
        y="Percent",
        color="Perception",
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        category_orders={"FY": year_labels, "Perception": PERCEPTION_ORDER},
        labels={"FY": "Fiscal Year", "Percent": "Percent of Responses"},
        title=title,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textfont_size=text_size, textposition="inside")
    return fig


def _combined_perception_chart(
    scores_df: pd.DataFrame,
    records: Iterable[dict[str, object]],
    *,
    years: Iterable[int],
    height: int = 420,
) -> go.Figure | None:
    record_list = [record for record in records]
    if not record_list:
        return None

    items = [record.get("item", {}) for record in record_list]
    if not items:
        return None

    question_ids: list[object] = []
    label_map: dict[object, str] = {}
    for record, item in zip(record_list, items):
        qid = item.get("QuestionID")
        if pd.isna(qid):
            continue
        question_ids.append(qid)
        label_map[qid] = record.get("label", str(qid))

    if not question_ids:
        return None

    year_list = list(years)
    subset = scores_df[
        (scores_df["QuestionID"].isin(question_ids)) & (scores_df["FY"].isin(year_list))
    ]
    if subset.empty:
        return None

    available_columns = [col for col in PERCEPTION_ORDER if col in subset.columns]
    if not available_columns:
        return None

    subset = subset[["QuestionID", "FY"] + available_columns]
    subset = subset.drop_duplicates(subset=["QuestionID", "FY"])
    if subset.empty:
        return None

    melted = subset.melt(
        id_vars=["QuestionID", "FY"],
        value_vars=available_columns,
        var_name="Perception",
        value_name="Percent",
    ).dropna(subset=["Percent"])

    if melted.empty:
        return None

    melted = melted.copy()
    melted["FY"] = melted["FY"].astype(int)
    melted["Percent"] = melted["Percent"].astype(float).round(2)
    melted["QuestionLabel"] = melted["QuestionID"].map(label_map).fillna(
        melted["QuestionID"].astype(str)
    )
    year_labels = [str(year) for year in year_list]
    question_numbers = [str(qid).strip() for qid in question_ids]

    melted["Fiscal Year"] = melted["FY"].astype(str)
    melted["QuestionNumber"] = melted["QuestionID"].apply(lambda q: str(q).strip())
    melted["QuestionNumber"] = pd.Categorical(
        melted["QuestionNumber"], categories=question_numbers, ordered=True
    )
    melted["Perception"] = pd.Categorical(
        melted["Perception"], categories=PERCEPTION_ORDER, ordered=True
    )
    melted = melted.sort_values(["QuestionNumber", "Fiscal Year", "Perception"])

    rows = max(1, math.ceil(len(question_numbers) / 3))
    calculated_height = max(height, rows * 320)

    fig = px.bar(
        melted,
        x="Fiscal Year",
        y="Percent",
        color="Perception",
        barmode="stack",
        facet_col="QuestionNumber",
        facet_col_wrap=3,
        color_discrete_map=COLOR_MAP,
        category_orders={
            "Fiscal Year": year_labels,
            "Perception": PERCEPTION_ORDER,
            "QuestionNumber": question_numbers,
        },
        labels={
            "Fiscal Year": "Fiscal Year",
            "Percent": "Percent of Responses",
        },
        hover_data={
            "QuestionLabel": True,
            "Fiscal Year": True,
            "Percent": ":.1f",
        },
    )

    fig.update_layout(
        height=calculated_height,
        margin=dict(l=10, r=10, t=40, b=60),
        title=None,
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 100], title="Percent of Responses", matches="y")
    fig.update_xaxes(
        title_text="",
        tickmode="array",
        tickvals=year_labels,
        ticktext=year_labels,
        showticklabels=True,
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="inside", textfont_size=11)

    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split("=")[-1].strip(),
            font=dict(size=16, color="#212529"),
        )
    )

    return fig


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


def _format_question_text(question_id: object, question_text: object) -> str:
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

strength_records = [
    {
        "label": _format_question_text(item["QuestionID"], item["QuestionText"]),
        "item": item,
    }
    for item in top_strengths
]

if not strength_records:
    st.info("No chartable data found for the top strengths.")
else:
    st.sidebar.subheader("Filters")
    option_labels = ["All"] + [record["label"] for record in strength_records]
    selected_label = st.sidebar.selectbox("Survey item", options=option_labels)
    _render_sidebar_legend()

    if selected_label == "All":
        st.subheader("Top Strength Items")
        combined_fig = _combined_perception_chart(
            scores,
            strength_records,
            years=years_to_show,
        )

        if combined_fig is None:
            st.info("Perception breakdown unavailable for the selected survey items.")
        else:
            st.plotly_chart(combined_fig, config=PLOTLY_CONFIG, use_container_width=True)
            summary_lines = []
            for record in strength_records:
                item = record["item"]
                avg_positive = item.get("AveragePositive")
                avg_text = f"{avg_positive:.2f}%" if avg_positive is not None else "N/A"
                summary_lines.append(
                    f"- **{record['label']}** — 3-Year Avg Positive: {avg_text}"
                )
            if summary_lines:
                st.markdown("\n".join(summary_lines))
    else:
        label_to_item = {record["label"]: record["item"] for record in strength_records}
        selected_item = label_to_item[selected_label]
        fig = _perception_chart(
            scores,
            selected_item["QuestionID"],
            years=years_to_show,
            title=selected_label,
        )
        if fig is None:
            st.info("Perception breakdown unavailable for the selected survey item.")
        else:
            avg_positive = selected_item.get("AveragePositive")
            detail_lines = [
                f"**3-Year Avg Positive:** {avg_positive:.2f}%" if avg_positive is not None else "**3-Year Avg Positive:** N/A"
            ]
            subindex_label = selected_item.get("SubIndex", "")
            if subindex_label:
                detail_lines.append(f"<span style='color: #6c757d;'>Sub-Index: {subindex_label}</span>")
            index_label = selected_item.get("Index")
            if index_label:
                detail_lines.append(f"<span style='color: #6c757d;'>Index: {index_label}</span>")

            st.markdown("<br/>".join(detail_lines), unsafe_allow_html=True)
            st.plotly_chart(fig, config=PLOTLY_CONFIG)


table_rows: list[dict[str, object]] = []
for item in top_strengths:
    per_year = item["PerYear"]
    row: dict[str, object] = {
        "Survey Item": _format_question_text(item["QuestionID"], item["QuestionText"]),
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
