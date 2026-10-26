
from pathlib import Path
from typing import Iterable

import pymupdf
from llama_index.core import Document


def load_pdfs(paths: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []

    for path in paths:
        pdf_path = Path(path)

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {pdf_path}")

        loaded = []
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    loaded.append(
                        Document(
                            text=text,
                            metadata={
                                "file_name": pdf_path.name,
                                "page_label": str(page_number),
                            },
                        )
                    )

        if not loaded:
            raise ValueError(
                f"No readable text found in {pdf_path}. "
                "The PDF may be scanned and require OCR."
            )

        documents.extend(loaded)

    if not documents:
        raise ValueError("No documents were loaded.")

    return documents


def load_directory(directory: str | Path) -> list[Document]:
    data_dir = Path(directory)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        raise ValueError(f"No PDF files found in {data_dir}")

    return load_pdfs(pdf_paths)
