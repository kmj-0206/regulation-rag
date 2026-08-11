from __future__ import annotations

from pathlib import Path

import pymupdf

from app.ocr import ocr_pdf_page


# 이 문자 수보다 직접 추출되는 글자가 적으면
# 이미지/스캔 페이지로 보고 OCR을 수행한다.
MIN_TEXT_LENGTH = 50


def normalize_text(text: str) -> str:
    """
    불필요한 공백과 빈 줄을 정리한다.
    """
    lines: list[str] = []

    for line in text.splitlines():
        cleaned = " ".join(
            line.split()
        )

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)


def extract_page_text(
    page: pymupdf.Page,
) -> tuple[str, str]:
    """
    PDF 페이지에서 텍스트를 추출한다.

    반환:
        (text, extraction_type)

    extraction_type:
        "text" 또는 "ocr"
    """

    direct_text = page.get_text(
        "text"
    )

    direct_text = normalize_text(
        direct_text
    )

    # 텍스트 PDF
    if len(direct_text) >= MIN_TEXT_LENGTH:
        return direct_text, "text"

    # 이미지 PDF / 스캔 페이지
    ocr_text = ocr_pdf_page(page)

    ocr_text = normalize_text(
        ocr_text
    )

    return ocr_text, "ocr"


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """
    PDF를 페이지별로 읽는다.

    텍스트가 있는 페이지는 직접 추출하고,
    텍스트가 없는 페이지는 OCR한다.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 파일이 없습니다: {pdf_path}"
        )

    document = pymupdf.open(
        str(pdf_path)
    )

    pages: list[dict] = []

    try:
        for page_index, page in enumerate(
            document,
            start=1,
        ):
            text, extraction_type = (
                extract_page_text(page)
            )

            print(
                f"[PDF] {pdf_path.name} "
                f"page={page_index} "
                f"type={extraction_type} "
                f"chars={len(text)}"
            )

            if not text:
                continue

            pages.append(
                {
                    "page": page_index,
                    "text": text,
                    "extraction_type": (
                        extraction_type
                    ),
                }
            )

    finally:
        document.close()

    return pages
