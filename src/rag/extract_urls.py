"""
extract_urls.py
===============
Reads the per-profession CSV files and returns URL records.

Each record is a dict::

    {
        "url":        str,
        "categories": List[str],   # all professions this URL is relevant to
        "source":     str,         # always "website" for web URLs
    }

The same URL may appear in multiple CSVs (e.g. a CDC page referenced by both
Nurse and PA programmes).  ``extract_all_urls()`` deduplicates by URL and
merges categories so each URL is fetched and stored exactly once.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

# Map CSV file path → category label
_SOURCES: Dict[str, str] = {
    "src/rag/Nurse.csv":         "Nurse",
    "src/rag/PA.csv":            "PA",
    "src/rag/Public_Health.csv": "Public_Health",
    "src/rag/Social_Work.csv":   "Social_Work",
}


def _extract_from_csv(csv_path: str, category: str) -> List[Dict[str, Any]]:
    """
    Return URL records from a single CSV where Type is 'website' and Needed is 'y'.

    Args:
        csv_path: Relative path to the CSV file.
        category: Category label to assign to each record.

    Returns:
        List of dicts with keys ``url``, ``categories``, ``source``.
    """
    records: List[Dict[str, Any]] = []
    rows = pd.read_csv(csv_path)
    for row in rows.itertuples():
        if (
            isinstance(row.Type, str)
            and isinstance(row.Needed, str)
            and row.Type.lower() == "website"
            and row.Needed.lower() == "y"
        ):
            records.append({
                "url":        row.URL,
                "categories": [category],
                "source":     "website",
            })
    return records


def extract_all_urls() -> List[Dict[str, Any]]:
    """
    Read all profession CSVs, deduplicate by URL, and merge categories.

    A URL that appears in both Nurse and PA CSVs will produce a single record
    with ``categories: ["Nurse", "PA"]`` rather than two separate records.

    Returns:
        List of unique URL records, each with ``url``, ``categories``, ``source``.
    """
    seen: Dict[str, Dict[str, Any]] = {}

    for csv_path, category in _SOURCES.items():
        for record in _extract_from_csv(csv_path, category):
            url = record["url"]
            if url in seen:
                if category not in seen[url]["categories"]:
                    seen[url]["categories"].append(category)
            else:
                seen[url] = record

    return list(seen.values())


# ── Per-profession helpers (kept for backward compatibility) ──────────────────

def extract_nurse() -> List[Dict[str, Any]]:
    """Return URL records for the Nurse programme."""
    return _extract_from_csv("src/rag/Nurse.csv", "Nurse")


def extract_pa() -> List[Dict[str, Any]]:
    """Return URL records for the PA programme."""
    return _extract_from_csv("src/rag/PA.csv", "PA")


def extract_public_health() -> List[Dict[str, Any]]:
    """Return URL records for the Public Health programme."""
    return _extract_from_csv("src/rag/Public_Health.csv", "Public_Health")


def extract_social_work() -> List[Dict[str, Any]]:
    """Return URL records for the Social Work programme."""
    return _extract_from_csv("src/rag/Social_Work.csv", "Social_Work")