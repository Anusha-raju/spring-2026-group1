from pdf_extractors.pdf_to_structured_text_xml import PDFtoXMLextractor
from pdf_extractors.pdfplumber import PDFPlumberExtractor
from pdf_extractors.pdf_pypdf import PDFPyPDF
from pathlib import Path

class ExtractPDF:
    def __init__(self):
        self.pdftoxmlextractor = PDFtoXMLextractor()
        self.pdfplumberextractor = PDFPlumberExtractor()
        self.pdfpypdfextractor = PDFPyPDF()
        pass

    def extract_pdf(self, filepath, method = None):
        result = ''
        if method == "xmlextractor":
            result = self.pdftoxmlextractor.pdf_extract(filepath)
        elif method == "pdfplumber":
            result = self.pdfplumberextractor.pdf_extract(filepath)
        elif method == "pypdf":
            result = self.pdfpypdfextractor.pdf_extract(filepath)
        
        return result



if __name__ == "__main__":
    pdfextractor = ExtractPDF()
    # print(pdfextractor.extract_pdf("src/rag/pdf_extractor/pdfs/a-drug-court-team-member's-guide-to-medication-in-addiction-treatment.pdf", method = "xmlextractor"))

    input_dir = Path("src/rag/pdf_extractor/pdfs")
    output_dir = Path("src/rag/pdf_extractor/outputs/pypdfmethod")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        print(pdf_file)
        extracted_text = pdfextractor.extract_pdf(
            str(pdf_file),
            method="pypdf"
        )

        output_file = output_dir / f"{pdf_file.stem}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print(f"Saved: {output_file}")
