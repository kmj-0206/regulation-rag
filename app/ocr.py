from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf
from paddleocr import PaddleOCR


_ocr: PaddleOCR | None = None


def get_ocr() -> PaddleOCR:
    """
    PaddleOCR 모델을 최초 한 번만 생성한다.
    """
    global _ocr

    if _ocr is None:
        _ocr = PaddleOCR(
            lang="korean",
        )

    return _ocr


def ocr_pdf_page(
    page: pymupdf.Page,
    dpi: int = 300,
) -> str:
    """
    PDF 페이지를 고해상도 이미지로 변환한 뒤 OCR한다.
    """

    pix = page.get_pixmap(
        dpi=dpi,
        alpha=False,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as tmp:
        image_path = Path(tmp.name)

    try:
        pix.save(str(image_path))

        ocr = get_ocr()

        results = ocr.predict(
            str(image_path)
        )

        texts: list[str] = []

        for result in results:
            data = result.json

            if callable(data):
                data = data()

            # PaddleOCR 버전에 따라 결과 구조가
            # 조금씩 다를 수 있으므로 안전하게 처리
            if not isinstance(data, dict):
                continue

            rec_texts = data.get(
                "rec_texts",
                [],
            )

            if isinstance(rec_texts, list):
                for text in rec_texts:
                    text = str(text).strip()

                    if text:
                        texts.append(text)

        return "\n".join(texts)

    finally:
        image_path.unlink(
            missing_ok=True
        )
