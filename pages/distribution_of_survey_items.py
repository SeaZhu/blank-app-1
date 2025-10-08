"""Distribution of Survey Items page."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

from fevs_io import load_excel
from fevs_processing import compute_question_scores, prepare_question_metadata


st.set_page_config(
    page_title="Distribution of Survey Items · FEVS-style Dashboard",
    layout="wide",
)


PLOTLY_CONFIG = {"displaylogo": False}
DEFAULT_PATH = Path("data/fevs_sample_data_3FYs_DataSet_5.xlsx")
COLOR_SEQUENCE = qualitative.Plotly


@st.cache_data(show_spinner=False)
def _load_excel_cached(fp: str) -> dict[str, pd.DataFrame]:
    return load_excel(fp)


def _kde_density(values: Iterable[float], grid: np.ndarray) -> np.ndarray:
    """Return a simple Gaussian KDE estimate for the provided values."""

    array = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if array.size == 0:
        return np.array([])

    if array.size == 1:
        bandwidth = 4.0
    else:
        std = array.std(ddof=1)
        bandwidth = 1.06 * std * (array.size ** (-1 / 5))
        bandwidth = float(max(bandwidth, 2.5))

    diffs = grid[:, None] - array[None, :]
    exponent = -0.5 * (diffs / bandwidth) ** 2
    density = np.exp(exponent).sum(axis=1)
    normaliser = bandwidth * np.sqrt(2 * np.pi) * array.size
    if normaliser == 0:
        return np.zeros_like(grid)
    return density / normaliser


def _assign_year_colors(years: Iterable[int]) -> dict[int, str]:
    colors = {}
    for idx, year in enumerate(sorted(years)):
        colors[year] = COLOR_SEQUENCE[idx % len(COLOR_SEQUENCE)]
    return colors


def _build_distribution_chart(
    df: pd.DataFrame,
    *,
    years: Iterable[int],
    colors: dict[int, str],
    title: str,
) -> go.Figure | None:
    grid = np.linspace(0, 100, 256)
    figure = go.Figure()

    added = False
    max_density = 0.0
    annotations: list[dict[str, object]] = []

    for idx, year in enumerate(sorted(years)):
        series = df.loc[df["FY"] == year, "Positive"].dropna()
        if series.empty:
            continue

        density = _kde_density(series.values, grid)
        if density.size == 0:
            continue

        color = colors.get(year, COLOR_SEQUENCE[idx % len(COLOR_SEQUENCE)])
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode="lines",
                name=f"FY{year}",
                line=dict(color=color, width=3),
            )
        )

        added = True
        max_density = max(max_density, float(density.max()))

        mean_value = float(series.mean())
        figure.add_vline(
            x=mean_value,
            line_color=color,
            line_dash="dash",
            line_width=2,
        )
        annotations.append(
            dict(
                x=mean_value,
                y=1.02 + 0.04 * idx,
                xref="x",
                yref="paper",
                text=f"FY{year}: {mean_value:.1f}%",
                showarrow=False,
                font=dict(color=color, size=12),
                align="left",
            )
        )

    if not added:
        return None

    figure.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(title="Positive Response (%)", range=[0, 100]),
        yaxis=dict(title="Density", rangemode="tozero"),
        annotations=annotations,
    )

    if max_density > 0:
        figure.update_yaxes(range=[0, max_density * 1.15])

    return figure


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
def_sheet = sheets.get("Index-Def-Map")

if raw is None or raw.empty:
    st.error("Sheet 'fevs_sample_data_3FYs_Set5' missing or empty.")
    st.stop()

metadata = prepare_question_metadata(map_sheet, def_sheet)
if metadata.empty:
    st.error("Could not derive question metadata from the workbook.")
    st.stop()

metadata = metadata[metadata["QuestionID"].isin(raw.columns)]
metadata["Performance Dimension"] = metadata["Performance Dimension"].fillna("").astype(str).str.strip()
metadata = metadata[metadata["Performance Dimension"].astype(str).str.strip() != ""]
metadata = metadata[metadata["Performance Dimension"].str.lower() != "other"]

if metadata.empty:
    st.info("No indexed survey items available after excluding 'Other'.")
    st.stop()

question_ids = metadata["QuestionID"].tolist()
scores = compute_question_scores(raw, question_ids)

if scores.empty:
    st.warning("No response data available for the selected workbook.")
    st.stop()

scores = scores.merge(
    metadata[["QuestionID", "QuestionText", "Performance Dimension", "Index", "SubIndex"]],
    on="QuestionID",
    how="left",
)
scores = scores.dropna(subset=["FY", "QuestionID"])
scores["FY"] = scores["FY"].astype(int)
scores["Positive"] = scores["Positive"].astype(float)
scores["Performance Dimension"] = scores["Performance Dimension"].fillna("").astype(str).str.strip()
scores = scores[scores["Performance Dimension"].astype(str).str.strip() != ""]
scores = scores[scores["Performance Dimension"].str.lower() != "other"]

available_years = sorted(scores["FY"].unique())
year_colors = _assign_year_colors(available_years)

index_options = ["All"] + sorted(
    idx for idx in scores["Performance Dimension"].unique() if idx
)

st.sidebar.header("Filters")
selected_index = st.sidebar.selectbox("Index", index_options, index_options.index("All"))

if selected_index != "All":
    filtered_scores = scores[scores["Performance Dimension"] == selected_index]
else:
    filtered_scores = scores

st.title("Distribution of Survey Items")
st.caption("Positive response distribution for each fiscal year with index filter support.")

if filtered_scores.empty:
    st.info("No survey items found for the selected filter.")
    st.stop()

chart = _build_distribution_chart(
    filtered_scores,
    years=available_years,
    colors=year_colors,
    title=(
        "Survey Item Positive Response Distribution"
        if selected_index == "All"
        else f"Survey Item Distribution · {selected_index}"
    ),
)

if chart is None:
    st.info("Not enough data to render the distribution plot for the selected filter.")
else:
    st.plotly_chart(chart, config=PLOTLY_CONFIG)
