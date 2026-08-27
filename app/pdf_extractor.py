# app/pdf_extractor.py

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.ocr import ocr_pdf_page


# 직접 추출되는 텍스트가 이보다 적으면
# 이미지/스캔 페이지로 보고 OCR 수행
MIN_TEXT_LENGTH = 20


def normalize_direct_text(
    text: str,
) -> str:
    """
    일반 텍스트 PDF 전용 정규화.

    PDF 내부의 줄바꿈, 탭, 연속 공백을
    모두 하나의 공백으로 통일한다.

    예:
        제1항
        각
        호의
        초과수혜

    ->
        제1항 각 호의 초과수혜
    """

    if not text:
        return ""

    return " ".join(
        text.split()
    ).strip()


def normalize_ocr_text(
    text: str,
) -> str:
    """
    OCR 결과 전용 정규화.

    OCR 결과는 [MAJOR], [ORGANIZATION],
    항목 | 전화번호 등의 줄 구조 자체가 중요하므로
    줄바꿈을 유지한다.
    """

    if not text:
        return ""

    lines: list[str] = []

    for line in text.splitlines():
        cleaned = " ".join(
            line.split()
        ).strip()

        if cleaned:
            lines.append(
                cleaned
            )

    return "\n".join(
        lines
    )


def extract_page_text(
    page: pymupdf.Page,
) -> tuple[str, str]:
    """
    PDF 페이지에서 텍스트를 추출한다.

    일반 PDF:
        PyMuPDF 직접 추출
        -> 공백/줄바꿈 단순 정규화

    이미지/스캔 PDF:
        OCR
        -> 줄 구조 유지

    반환:
        (text, extraction_type)

    extraction_type:
        "text"
        "ocr"
    """

    # 먼저 직접 텍스트 존재 여부 확인
    raw_text = page.get_text(
        "text"
    )

    normalized_direct = (
        normalize_direct_text(
            raw_text
        )
    )

    # 일반 텍스트 PDF
    if (
        len(normalized_direct)
        >= MIN_TEXT_LENGTH
    ):
        return (
            normalized_direct,
            "text",
        )

    # 이미지 / 스캔 PDF
    ocr_text = ocr_pdf_page(
        page
    )

    normalized_ocr = (
        normalize_ocr_text(
            ocr_text
        )
    )

    return (
        normalized_ocr,
        "ocr",
    )


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """
    PDF를 페이지별로 읽는다.

    일반 텍스트 PDF:
        직접 추출

    이미지/스캔 PDF:
        OCR
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 파일이 없습니다: "
            f"{pdf_path}"
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
            try:
                text, extraction_type = (
                    extract_page_text(
                        page
                    )
                )

            except Exception as exc:
                # 한 페이지의 OCR/추출 실패가
                # 문서 전체 처리를 막지 않도록 건너뛴다.
                print(
                    f"[PDF] "
                    f"{pdf_path.name} "
                    f"page={page_index} "
                    f"추출 실패: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            print(
                f"[PDF] "
                f"{pdf_path.name} "
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
