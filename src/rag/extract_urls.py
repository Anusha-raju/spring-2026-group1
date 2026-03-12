"""
Reads website_knowledge.csv and returns URL records for web extraction.
"""

from __future__ import annotations
from typing import Any, Dict, List
import csv
import os

_WEBSITE_KNOWLEDGE_CSV = os.path.join(os.path.dirname(__file__), "website_knowledge.csv")


def extract_all_urls() -> List[Dict[str, Any]]:
    """
    Read website_knowledge.csv, deduplicate by URL, and merge categories.

    A URL that appears multiple times will produce a single record with all
    categories merged, e.g. ``categories: ["Nurse", "Physician Assistant"]``.

    Returns:
        List of unique URL records, each with ``url``, ``categories``, ``source``.
    """
    seen: Dict[str, Dict[str, Any]] = {}

    with open(_WEBSITE_KNOWLEDGE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("Web URL", "").strip()
            role_str = row.get("Role", "").strip()
            if not url or not role_str:
                continue
            roles = [r.strip() for r in role_str.split(",") if r.strip()]
            if url in seen:
                for role in roles:
                    if role not in seen[url]["categories"]:
                        seen[url]["categories"].append(role)
            else:
                seen[url] = {"url": url, "categories": roles, "source": "website"}

    return list(seen.values())
