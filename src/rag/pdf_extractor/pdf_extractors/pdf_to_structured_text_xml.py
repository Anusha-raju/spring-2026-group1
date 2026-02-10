from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import io
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List


class PDFtoXMLextractor:
    def __init__(self):
        pass

    def pdf_to_xml(self, filename):
        output = io.BytesIO()
        with open(filename, "rb") as f:
            extract_text_to_fp(f, output, laparams=LAParams(), output_type='xml')
        return output.getvalue()
    
    def xml_content_formattor(self, xmloutput, para_join: str = "\n",
                                         collapse_spaces: bool = True) -> Dict[str, Any]:
                                         
        """
        Reads your pdfminer-style XML:

        Output:
        {
            "pages": [
            {
                "page_id": "1",
                "page_bbox": "...",
                "textboxes": {
                "0": {
                    "textbox_bbox": "...",
                    "lines": [...],
                    "paragraph": "..."
                },
                ...
                }
            },
            ...
            ]
        }
        """
        # root = ET.parse(xml_path).getroot()
        root = ET.fromstring(xmloutput)
        out_pages: List[Dict[str, Any]] = []

        for page in root.findall("./page"): 
            page_id = page.get("id")
            page_bbox = page.get("bbox")

            tb_map: Dict[str, Dict[str, Any]] = {}

            for tb in page.findall("./textbox"):
                tb_id = tb.get("id")
                tb_bbox = tb.get("bbox")

                lines: List[str] = []

                for tl in tb.findall("./textline"):
                    parts = [(t.text or "") for t in tl.findall("./text")]

                    line = "".join(parts)

                    if collapse_spaces:
                        line = re.sub(r"\s+", " ", line).strip()
                    else:
                        line = line.strip()

                    if line:
                        lines.append(line)

                paragraph = para_join.join(lines).strip()

                tb_map[tb_id] = {
                    "textbox_bbox": tb_bbox,
                    "lines": lines,
                    "paragraph": paragraph
                }

            out_pages.append({
                "page_id": page_id,
                "page_bbox": page_bbox,
                "textboxes": tb_map
            })

        return {"pages": out_pages}
    



    def format_pages_from_textboxes(self,
        result: Dict[str, Any],
        *,
        collapse_internal_newlines: bool = True,
    ) -> Dict[str, Any]:
        """
        Takes pdf_to_textboxes() result and returns:
        {
            "pages_text": [{"page_id": "...", "text": "..."}, ...],
            "full_text": "...\n\n..."
        }

        Rules while combining textbox paragraphs (in id order):
        1) Replace internal '\n' in each paragraph with a single space.
        2) If the previous paragraph ends with '.' then insert '\n' before next paragraph.
        3) Else if the next paragraph starts with a Capital letter then insert '\n' before it.
        4) Else combine with a space.
        """
        pages_text: List[Dict[str, str]] = []

        def normalize_para(p: str) -> str:
            p = (p or "").strip()
            if collapse_internal_newlines:
                p = p.replace("\n", " ")
            p = re.sub(r"\s+", " ", p).strip()
            return p

        def starts_with_capital(s: str) -> bool:
            # Finds first alphabetic character; checks if it's uppercase
            m = re.search(r"[A-Za-z]", s)
            return bool(m) and m.group(0).isupper()

        for page in result.get("pages", []):
            textboxes = page.get("textboxes", {}) or {}

            # pdfminer textbox ids are strings like "0","1","2"...
            # Sort numerically when possible.
            def sort_key(k: str):
                try:
                    return int(k)
                except Exception:
                    return k

            tb_ids = sorted(textboxes.keys(), key=sort_key)

            combined_parts: List[str] = []
            prev_para = ""

            for tb_id in tb_ids:
                para_raw = textboxes[tb_id].get("paragraph", "")
                para = normalize_para(para_raw)
                if not para:
                    continue

                if not combined_parts:
                    combined_parts.append(para)
                    prev_para = para
                    continue

                prev_ends_with_period = prev_para.rstrip().endswith(".")
                next_starts_cap_not_allcaps = starts_with_capital(para)

                if prev_ends_with_period or next_starts_cap_not_allcaps:
                    combined_parts.append("\n" + para)
                else:
                    combined_parts.append(" " + para)

                prev_para = para

            page_text = "".join(combined_parts).strip()
            pages_text.append({"page_id": str(page.get("page_id", "")), "text": page_text})

        full_text = "\n\n".join(p["text"] for p in pages_text if p["text"]).strip()
        return {"pages_text": pages_text, "full_text": full_text}

    def pdf_extract(self, filename):
        xmloutput = self.pdf_to_xml(filename)
        result = self.xml_content_formattor(xmloutput)
        pdf_text = self.format_pages_from_textboxes(result)
        return pdf_text['full_text']
