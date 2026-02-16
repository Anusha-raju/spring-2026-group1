import os
from pathlib import Path
import fitz 
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

class PDFDeepSeek:
    def __init__(self, device = "cpu"):
        self.tokenizer, self.model = self.load_deepseek_ocr(device=device)
        pass
    def pdf_to_images_pymupdf(self, pdf_path: str, out_dir: str, dpi: int = 200) -> list[str]:
        """
        Renders each PDF page to a PNG using PyMuPDF.
        Returns a list of image file paths in page order.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0 
        mat = fitz.Matrix(zoom, zoom)

        image_paths = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = os.path.join(out_dir, f"page_{i+1:04d}.png")
            pix.save(img_path)
            image_paths.append(img_path)

        return image_paths


    def load_deepseek_ocr(self, device: str = "cuda"):
        model_name = "deepseek-ai/DeepSeek-OCR"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
        )

        model = model.eval()

        if device == "cuda":
            model = model.cuda().to(torch.bfloat16)
        elif device == "cpu":
            model = model.cpu()
        else:
            raise ValueError("device must be 'cuda' or 'cpu'")

        return tokenizer, model


    def ocr_images(self, tokenizer, model, image_paths: list[str], out_dir: str, prompt: str):
        """
        Runs DeepSeek-OCR on each image and writes per-page markdown/text.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        per_page_outputs = []
        for idx, img_path in enumerate(image_paths, start=1):
           
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=img_path,
                output_path=out_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True,
            )

            text = res if isinstance(res, str) else str(res)

            page_out = os.path.join(out_dir, f"page_{idx:04d}.md")
            with open(page_out, "w", encoding="utf-8") as f:
                f.write(text)

            per_page_outputs.append(text)

        combined_path = os.path.join(out_dir, "combined.md")
        with open(combined_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(per_page_outputs, start=1):
                f.write(f"\n\n<!-- PAGE {i} -->\n\n")
                f.write(t)

        return combined_path


    def pdf_extract(self, filepath, device = "cpu"):
            workdir = os.path.abspath('out')
            prompt="<image>\n<|grounding|>Convert the document to markdown."
            images_dir = os.path.join(workdir, "pages")
            ocr_dir = os.path.join(workdir, "ocr")
            image_paths = self.pdf_to_images_pymupdf(filepath, images_dir, dpi=220)
            combined = self.ocr_images(self.tokenizer, self.model, image_paths, ocr_dir, prompt=prompt)
            print(f"Done. Combined output: {combined}")
