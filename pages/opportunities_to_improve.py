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


def _format_question_text(question_id: str, question_text: str) -> str:
    question_text = (question_text or "").strip()
    if not question_text:
        question_text = "Question text unavailable"
    question_id = str(question_id).strip() if question_id is not None else ""
    if question_id:
        return f"{question_id}. {question_text}"
    return question_text


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

question_options = {
    _format_question_text(row["QuestionID"], row["QuestionText"]): row
    for row in lowest_five.to_dict("records")
}

if not question_options:
    st.info("No chartable data available for the lowest survey items.")
else:
    selected_label = st.selectbox(
        "Select a survey item to view perception trends",
        options=list(question_options.keys()),
    )
    selected_row = question_options[selected_label]
    selected_scores = recent_scores[
        (recent_scores["QuestionID"] == selected_row["QuestionID"])
        & (recent_scores["FY"].isin(years_to_show))
    ].copy()

    if selected_scores.empty:
        st.info("No perception data available for the selected survey item.")
    else:
        selected_scores["FY"] = selected_scores["FY"].astype(int).astype(str)
        perception_columns = ["Positive", "Neutral", "Negative"]
        perception_df = selected_scores[["FY"] + [col for col in perception_columns if col in selected_scores.columns]]
        perception_df = perception_df.drop_duplicates(subset=["FY"])

        melted = perception_df.melt(
            id_vars="FY",
            value_vars=perception_columns,
            var_name="Perception",
            value_name="Percent",
        ).dropna(subset=["Percent"])

        if melted.empty:
            st.info("Perception breakdown unavailable for the selected survey item.")
        else:
            melted["FY"] = pd.Categorical(
                melted["FY"],
                categories=[str(year) for year in years_to_show],
                ordered=True,
            )
            melted["Percent"] = melted["Percent"].astype(float).round(2)
            melted["Perception"] = pd.Categorical(
                melted["Perception"],
                categories=["Positive", "Neutral", "Negative"],
                ordered=True,
            )

            color_map = {
                "Positive": "#0B5ED7",
                "Neutral": "#6C757D",
                "Negative": "#E5533D",
            }

            fig = px.bar(
                melted,
                x="FY",
                y="Percent",
                color="Perception",
                barmode="stack",
                color_discrete_map=color_map,
                category_orders={"FY": [str(year) for year in years_to_show], "Perception": ["Positive", "Neutral", "Negative"]},
                labels={"FY": "Fiscal Year", "Percent": "Percent of Responses"},
                title=selected_label,
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=60, b=10),
                legend_title="Perception",
                yaxis=dict(range=[0, 100]),
            )
            fig.update_traces(texttemplate="%{y:.2f}%", textfont_size=14, textposition="inside")

            avg_positive = selected_row.get("ThreeYearAverage")
            detail_lines = [
                f"**3-Year Avg Positive:** {avg_positive:.2f}%" if pd.notna(avg_positive) else "**3-Year Avg Positive:** N/A"
            ]
            index_label = selected_row.get("Performance Dimension", "")
            if index_label:
                detail_lines.append(f"<span style='color: #6c757d;'>Index: {index_label}</span>")
            subindex_label = selected_row.get("SubIndex", "")
            if subindex_label:
                detail_lines.append(f"<span style='color: #6c757d;'>Sub-Index: {subindex_label if subindex_label else 'Ungrouped Items'}</span>")

            st.markdown("<br/>".join(detail_lines), unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

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

