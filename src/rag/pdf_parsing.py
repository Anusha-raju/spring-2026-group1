"""
- PyMuPDF (fitz) for text and structure
- pdfplumber for tables
"""

import fitz
import pdfplumber
import re
import json
import os
from typing import List, Dict
from pathlib import Path


class HybridPDFParser:
    def __init__(self):
        pass

    def parse(self, pdf_path: str) -> Dict:
        """Parse PDF using hybrid approach"""
        print(f"Parsing: {pdf_path}")

        # Open with both libraries
        fitz_doc = fitz.open(pdf_path)
        plumber_doc = pdfplumber.open(pdf_path)

        structured_doc = []
        current_section = None
        all_tables = []

        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            plumber_page = plumber_doc.pages[page_num]

            print(f"Processing page {page_num + 1}/{len(fitz_doc)}")

            # Extract tables using pdfplumber
            page_tables = self._extract_tables_pdfplumber(
                plumber_page, page_num + 1)

            # Get table bounding boxes to avoid duplicating text
            table_bboxes = [t["bbox"] for t in page_tables if "bbox" in t]

            # Extract text using PyMuPDF
            blocks = fitz_page.get_text("blocks")
            blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))

            for block in blocks_sorted:
                x0, y0, x1, y1, raw_text, *_ = block

                # Skip blocks that overlap with tables
                if self._overlaps_with_tables((x0, y0, x1, y1), table_bboxes):
                    continue

                text = self.clean_text(raw_text)

                if not text or self.is_noise(text):
                    continue

                block_type = self.detect_type(text)
                indent = self.detect_indent(x0)

                # Create new section on heading
                if block_type == "heading":
                    current_section = {
                        "section": text,
                        "page": page_num + 1,
                        "content": []
                    }
                    structured_doc.append(current_section)

                # Add content to section
                elif current_section is not None:
                    current_section["content"].append({
                        "type": block_type,
                        "indent": indent,
                        "text": text,
                        "page": page_num + 1
                    })

            # Add tables to current section
            if current_section and page_tables:
                for table in page_tables:
                    current_section["content"].append(table)
                    all_tables.append(table)

        # Close documents
        fitz_doc.close()
        plumber_doc.close()

        # Merge adjacent headings
        structured_doc = self._merge_headings(structured_doc)

        return {
            "sections": structured_doc,
            "statistics": {
                "total_sections": len(structured_doc),
                "total_tables": len(all_tables)
            }
        }

    def _extract_tables_pdfplumber(
        self,
        page: pdfplumber.page.Page,
        page_num: int
    ) -> List[Dict]:
        """Extract tables using pdfplumber"""
        tables = []

        # pdfplumber's table extraction settings
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 3,
            "min_words_horizontal": 1,
            "intersection_tolerance": 3,
        }

        extracted_tables = page.extract_tables(table_settings)

        for i, table_data in enumerate(extracted_tables):
            # Need at least header + 1 row
            if not table_data or len(table_data) < 2:
                continue

            # Clean table data
            cleaned_table = []
            for row in table_data:
                cleaned_row = [
                    self.clean_text(cell) if cell else ""
                    for cell in row
                ]
                # Only add non-empty rows
                if any(cell.strip() for cell in cleaned_row):
                    cleaned_table.append(cleaned_row)

            if len(cleaned_table) >= 2:  # Has header + data
                # Try to find table bbox for text overlap detection
                bbox = page.bbox  # Default to page bbox
                try:
                    # Get approximate bbox from table coordinates
                    table_obj = page.find_tables(table_settings)
                    if i < len(table_obj):
                        bbox = table_obj[i].bbox
                except:
                    pass

                tables.append({
                    "type": "table",
                    "table_id": f"page{page_num}_table{i+1}",
                    "data": cleaned_table,
                    "rows": len(cleaned_table),
                    "cols": len(cleaned_table[0]) if cleaned_table else 0,
                    "page": page_num,
                    "bbox": bbox
                })

        return tables

    def _overlaps_with_tables(
        self,
        block_bbox: tuple,
        table_bboxes: List[tuple],
        threshold: float = 0.5
    ) -> bool:
        """Check if block overlaps significantly with any table"""
        bx0, by0, bx1, by1 = block_bbox
        block_area = (bx1 - bx0) * (by1 - by0)

        if block_area == 0:
            return False

        for tx0, ty0, tx1, ty1 in table_bboxes:
            # Calculate intersection
            ix0 = max(bx0, tx0)
            iy0 = max(by0, ty0)
            ix1 = min(bx1, tx1)
            iy1 = min(by1, ty1)

            if ix0 < ix1 and iy0 < iy1:
                intersection_area = (ix1 - ix0) * (iy1 - iy0)
                overlap_ratio = intersection_area / block_area

                if overlap_ratio > threshold:
                    return True

        return False

    # Text processing methods (from your original code)
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r"--- Page \d+ ---", "", text)
        text = re.sub(
            r"\d+\s*•\s*©\d{4}\s+ASAM and NADCP\s+All rights reserved\.?",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"©\d{4}\s+ASAM and NADCP\s+All rights reserved\.?\s*•?\s*\d*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r"•\s+", "• ", text)
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def is_noise(text: str) -> bool:
        """Detect if text is noise"""
        t = text.lower().strip()
        if len(t) < 2:
            return True
        if "all rights reserved" in t:
            return True
        if re.match(r'^\d+$', t):
            return True
        return False

    @staticmethod
    def detect_type(text: str) -> str:
        """Detect block type"""
        t = text.strip()

        if re.match(r'^(principle|figure|table|note)', t.lower()):
            return "paragraph"
        if re.match(r'^\d+\.\s+', t):
            return "step"
        if t.startswith(("•", "-", "*")):
            return "bullet"

        words = t.split()
        if (
            len(words) <= 10
            and not t.endswith(".")
            and sum(w[0].isupper() for w in words if w and w[0].isalpha()) >= max(1, len(words) * 0.6)
        ):
            return "heading"

        return "paragraph"

    @staticmethod
    def detect_indent(x: float) -> str:
        """Detect indentation level"""
        if x < 35:
            return "level_1"
        elif x < 70:
            return "level_2"
        else:
            return "level_3"

    @staticmethod
    def _merge_headings(sections: List[Dict]) -> List[Dict]:
        """Merge adjacent headings"""
        if not sections:
            return sections

        merged = []
        buffer = None

        for sec in sections:
            if buffer is None:
                buffer = sec
                continue

            if (
                len(buffer["content"]) == 0
                and len(sec["content"]) == 0
                and buffer["page"] == sec["page"]
            ):
                buffer["section"] += " " + sec["section"]
            else:
                merged.append(buffer)
                buffer = sec

        if buffer:
            merged.append(buffer)

        return merged

    def save_json(self, result: Dict, output_path: str):
        """Save to JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved JSON: {output_path}")

    def save_markdown(self, result: Dict, output_path: str, chunk_output_path: str | None = None):
        """Save to Markdown with proper table formatting"""
        md_lines = []

        md_lines.append("# Document\n\n")

        for section in result["sections"]:
            md_lines.append(f"## {section['section']}\n\n")

            for item in section["content"]:
                if item["type"] == "table":
                    md_lines.append(self._format_table_markdown(item))
                    md_lines.append("\n")

                elif item["type"] == "bullet":
                    md_lines.append(f"- {item['text']}\n")

                elif item["type"] == "step":
                    md_lines.append(f"{item['text']}\n\n")

                else:
                    md_lines.append(f"{item['text']}\n\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(md_lines)

        print(f"✅ Saved Markdown: {output_path}")
        if chunk_output_path is None:
            base, _ = os.path.splitext(output_path)
            chunk_output_path = f"{base}_chunks.jsonl"
        try:
            from chunk_markdown import chunk_markdown_file
            chunk_markdown_file(output_path, chunk_output_path)
        except Exception as e:
            print(f"⚠️  Chunking Markdown failed: {e}")

    @staticmethod
    def _format_table_markdown(table: Dict) -> str:
        """Format table as markdown"""
        if not table.get("data"):
            return ""

        lines = ["\n"]

        # Header row
        header = table["data"][0]
        lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Data rows
        for row in table["data"][1:]:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        lines.append("")
        return "\n".join(lines)
