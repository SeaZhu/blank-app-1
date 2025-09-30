"""Shared data preparation helpers for FEVS pages."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from fevs_calculations import likert_split


def prepare_question_metadata(map_df: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
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


def compute_question_scores(raw: pd.DataFrame, question_ids: Iterable[str]) -> pd.DataFrame:
    """Calculate perception shares for each question and fiscal year."""

    records: list[dict[str, object]] = []
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["FY", "QuestionID", "Positive", "Neutral", "Negative", "Responses"]
        )

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

