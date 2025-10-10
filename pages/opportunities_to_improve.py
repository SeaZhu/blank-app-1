"""Opportunities to Improve page."""
from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata


st.set_page_config(
    page_title="Opportunities to Improve · FEVS-style Dashboard",
    layout="wide",
)


PERCEPTION_ORDER = ["Positive", "Neutral", "Negative"]
COLOR_MAP = {
    "Positive": "#0B5ED7",
    "Neutral": "#6C757D",
    "Negative": "#E5533D",
}
PLOTLY_CONFIG = {"displaylogo": False}


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


def _compact_title(
    question_id: object,
    question_text: object,
    *,
    width: int = 58,
    max_lines: int | None = 3,
) -> str:
    base_text = ""
    if isinstance(question_text, str):
        base_text = question_text.strip()
    if not base_text:
        base_text = "Question text unavailable"

    prefix = ""
    if pd.notna(question_id) and str(question_id).strip():
        prefix = f"{str(question_id).strip()}. "

    available_width = max(width - len(prefix), 12)
    wrapped_lines = textwrap.wrap(
        base_text,
        width=available_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped_lines:
        wrapped_lines = [base_text]

    if max_lines is not None and len(wrapped_lines) > max_lines:
        preserved = wrapped_lines[: max_lines - 1] if max_lines > 1 else []
        remainder = " ".join(wrapped_lines[max_lines - 1 :])
        shortened = textwrap.shorten(remainder, width=available_width, placeholder="…")
        wrapped_lines = preserved + [shortened]

    if wrapped_lines:
        wrapped_lines[0] = prefix + wrapped_lines[0]
    else:
        wrapped_lines = [prefix.rstrip()]

    return "<br>".join(wrapped_lines)


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
        margin=dict(l=10, r=10, t=90, b=10),
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
            pad=dict(t=20),
        ),
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textfont_size=text_size, textposition="inside")
    return fig


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


def _format_question_text(question_id: object, question_text: object) -> str:
    question_text = (question_text or "").strip()
    if not question_text:
        question_text = "Question text unavailable"
    question_id = str(question_id).strip() if question_id is not None else ""
    if question_id:
        return f"{question_id}. {question_text}"
    return question_text


DEFAULT_PATH = Path("data/fevs_sample_data_3FYs_DataSet_5.xlsx")

title_placeholder = st.empty()

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

if metadata.empty:
    st.error("Could not derive question metadata from the workbook.")
    st.stop()

metadata = metadata[metadata["QuestionID"].isin(raw.columns)]
metadata["SubIndex"] = metadata["SubIndex"].fillna("").astype(str).str.strip()
metadata["Performance Dimension"] = (
    metadata["Performance Dimension"].fillna("").astype(str).str.strip()
)

metadata = metadata[metadata["Performance Dimension"].astype(str).str.strip() != ""]
metadata = metadata[metadata["Performance Dimension"].str.lower() != "other"]

if metadata.empty:
    st.info("No indexed survey items available after excluding 'Other'.")
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
scores["Performance Dimension"] = scores["Performance Dimension"].fillna("").astype(str).str.strip()
scores = scores[scores["Performance Dimension"].astype(str).str.strip() != ""]
scores = scores[scores["Performance Dimension"].str.lower() != "other"]
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

question_records = [
    {
        "label": _format_question_text(row["QuestionID"], row["QuestionText"]),
        "row": row,
    }
    for row in lowest_five.to_dict("records")
]

if not question_records:
    st.info("No chartable data available for the lowest survey items.")
else:
    st.sidebar.subheader("Filters")
    option_labels = ["All"] + [record["label"] for record in question_records]
    selected_label = st.sidebar.selectbox("Survey item", options=option_labels)
    _render_sidebar_legend()

    if selected_label == "All":
        st.subheader("Lowest Items Comparison")
        charts: list[tuple[dict[str, object], go.Figure]] = []
        for record in question_records:
            chart_title = _compact_title(
                record["row"]["QuestionID"],
                record["row"]["QuestionText"],
                width=44,
                max_lines=None,
            )
            fig = _perception_chart(
                recent_scores,
                record["row"]["QuestionID"],
                years=years_to_show,
                title=chart_title,
                text_size=11,
                height=360,
            )
            if fig is not None:
                charts.append((record["row"], fig))

        if not charts:
            st.info("Perception breakdown unavailable for the selected survey items.")
        else:
            columns = st.columns(len(charts))
            for column, (row, fig) in zip(columns, charts):
                column.plotly_chart(fig, config=PLOTLY_CONFIG)
                avg_positive = row.get("ThreeYearAverage")
                if pd.notna(avg_positive):
                    column.caption(f"3-Year Avg Positive: {avg_positive:.2f}%")
                else:
                    column.caption("3-Year Avg Positive: N/A")
    else:
        label_to_row = {record["label"]: record["row"] for record in question_records}
        selected_row = label_to_row[selected_label]
        fig = _perception_chart(
            recent_scores,
            selected_row["QuestionID"],
            years=years_to_show,
            title=selected_label,
        )
        if fig is None:
            st.info("Perception data unavailable for the selected survey item.")
        else:
            avg_positive = selected_row.get("ThreeYearAverage")
            detail_lines = [
                f"**3-Year Avg Positive:** {avg_positive:.2f}%" if pd.notna(avg_positive) else "**3-Year Avg Positive:** N/A"
            ]
            index_label = selected_row.get("Performance Dimension", "")
            if index_label:
                detail_lines.append(f"<span style='color: #6c757d;'>Index: {index_label}</span>")
            subindex_label = selected_row.get("SubIndex", "")
            if subindex_label:
                detail_lines.append(f"<span style='color: #6c757d;'>Sub-Index: {subindex_label}</span>")

            st.markdown("<br/>".join(detail_lines), unsafe_allow_html=True)
            st.plotly_chart(fig, config=PLOTLY_CONFIG)

# Build summary table
yearly = (
    recent_scores[recent_scores["QuestionID"].isin(lowest_ids)]
    .pivot_table(index="QuestionID", columns="FY", values="Positive")
    .rename(columns=lambda c: str(int(c)))
)

summary = lowest_five.merge(yearly, on="QuestionID", how="left")


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
