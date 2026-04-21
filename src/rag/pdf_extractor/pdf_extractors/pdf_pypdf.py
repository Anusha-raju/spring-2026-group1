from pypdf import PdfReader


class PDFPyPDF:
    def __init__(self):
        pass

    def pdf_extract(self, filepath):
        reader = PdfReader(filepath)
        all_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

        final_text = "\n".join(all_text)
        return final_text
