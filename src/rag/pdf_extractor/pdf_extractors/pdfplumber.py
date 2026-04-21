from typing import List, Tuple, Dict, Any, Optional
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import pdfplumber


class PDFPlumberExtractor:
    def __init__(self):
        pass

    def _bbox_contains(self, bbox: Tuple[float, float, float, float], x: float, top: float, x_tol: float = 1.0, y_tol: float = 1.0) -> bool:
        """
        bbox = (x0, top, x1, bottom) in pdfplumber coords
        """
        x0, t0, x1, b1 = bbox
        return (x0 - x_tol) <= x <= (x1 + x_tol) and (t0 - y_tol) <= top <= (b1 + y_tol)

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """
        Convert a pdfplumber table (list of rows) to Markdown.
        """
        if not table or not table[0]:
            return ""

        def norm(cell: Optional[str]) -> str:
            return (cell or "").replace("\n", " ").strip()

        rows = [[norm(c) for c in row] for row in table]
        ncols = max(len(r) for r in rows)
        rows = [r + [""] * (ncols - len(r)) for r in rows]

        header = rows[0]
        sep = ["---"] * ncols
        body = rows[1:] if len(rows) > 1 else []

        def md_row(r: List[str]) -> str:
            return "| " + " | ".join(r) + " |"

        out = [md_row(header), md_row(sep)]
        out += [md_row(r) for r in body]
        return "\n".join(out)

    def extract_text_without_table_duplicates(self,
        page,
        table_settings: Dict[str, Any],
        x_tol: float = 2.0,
        y_tol: float = 2.0
    ) -> Tuple[str, List[List[List[Optional[str]]]]]:
        """
        Returns:
        - text_outside_tables (string): text excluding anything inside detected table regions
        - tables (list): structured tables from extract_tables()

        How it works:
        1) find_tables() -> table bboxes
        2) extract_words() -> filter words NOT in any table bbox
        3) rebuild "outside text" in a simple reading order
        4) extract_tables() -> keep structured table output
        """
        # 1) Get table bboxes (more reliable for dedup than trying to diff strings)
        table_bboxes: List[Tuple[float, float, float, float]] = []
        try:
            found = page.find_tables(table_settings=table_settings)
            table_bboxes = [t.bbox for t in found]  # (x0, top, x1, bottom)
        except Exception:
            table_bboxes = []

        # 2) Extract words with positions
        words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True  # better reading order when there are many text boxes
        )

        # 3) Filter out words that fall inside any table bbox
        outside_words = []
        for w in words:
            x = float(w["x0"])
            top = float(w["top"])
            in_table = any(self._bbox_contains(bb, x, top, x_tol=x_tol, y_tol=y_tol) for bb in table_bboxes)
            if not in_table:
                outside_words.append(w)

        # 4) Rebuild text outside tables (simple line grouping by "top")
        #    (You can make this fancier later; this is robust and readable.)
        outside_words.sort(key=lambda w: (w["top"], w["x0"]))

        lines: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_top: Optional[float] = None
        line_tol = 3.0  # adjust if your PDF has different font sizes

        for w in outside_words:
            if current_top is None or abs(float(w["top"]) - current_top) <= line_tol:
                current.append(w)
                current_top = float(w["top"]) if current_top is None else current_top
            else:
                lines.append(current)
                current = [w]
                current_top = float(w["top"])
        if current:
            lines.append(current)

        text_lines = []
        for ln in lines:
            ln.sort(key=lambda w: w["x0"])
            text_lines.append(" ".join(w["text"] for w in ln).strip())

        text_outside_tables = "\n".join([t for t in text_lines if t])

        # 5) Structured tables
        tables = page.extract_tables(table_settings=table_settings) or []

        return text_outside_tables, tables

    def pdf_extract(self, filepath):
         table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 3,
                "min_words_vertical": 3,
                "min_words_horizontal": 1,
                "intersection_tolerance": 3,
                "text_tolerance": 3
            }
         with pdfplumber.open(filepath) as pdf:
            chunks = []
            for i, page in enumerate(pdf.pages, start=1):
                page_text, tables = self.extract_text_without_table_duplicates(page, table_settings)

                chunks.append(f"\n\n=== Page {i} ===\n")
                if page_text.strip():
                    chunks.append(page_text.strip())

                for ti, tbl in enumerate(tables, start=1):
                    md = self._table_to_markdown(tbl)
                    if md.strip():
                        chunks.append(f"\n\n[Table {ti}]\n{md}")

            return "\n".join(chunks)

    

