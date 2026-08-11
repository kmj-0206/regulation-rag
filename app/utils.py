from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from app.pdf_extractor import (
    extract_pdf_pages,
)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    긴 텍스트를 청크로 분할한다.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size는 0보다 커야 합니다."
        )

    if overlap < 0:
        raise ValueError(
            "overlap은 0 이상이어야 합니다."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap은 chunk_size보다 작아야 합니다."
        )

    chunks: list[str] = []

    text_length = len(text)
    start = 0

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        if end < text_length:
            search_start = (
                start
                + int(chunk_size * 0.6)
            )

            candidates = [
                text.rfind(
                    "\n",
                    search_start,
                    end,
                ),
                text.rfind(
                    ". ",
                    search_start,
                    end,
                ),
                text.rfind(
                    "다. ",
                    search_start,
                    end,
                ),
                text.rfind(
                    "조 ",
                    search_start,
                    end,
                ),
            ]

            split_position = max(
                candidates
            )

            if split_position > start:
                end = (
                    split_position + 1
                )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = (
            end - overlap
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunk_id(
    source: str,
    page: int,
    chunk_index: int,
    text: str,
) -> str:
    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def extract_pdf_chunks(
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """
    텍스트 PDF와 이미지 PDF를
    자동으로 처리해서 청크를 생성한다.
    """

    print(
        f"\n[PDF 읽기] {pdf_path.name}"
    )

    pages = extract_pdf_pages(
        pdf_path
    )

    records: list[
        dict[str, Any]
    ] = []

    for page_data in pages:
        page_number = int(
            page_data["page"]
        )

        text = str(
            page_data["text"]
        )

        extraction_type = str(
            page_data[
                "extraction_type"
            ]
        )

        page_chunks = split_text(
            text
        )

        print(
            f"  - {page_number}페이지: "
            f"{len(text):,}자, "
            f"{len(page_chunks)}개 청크, "
            f"{extraction_type}"
        )

        for chunk_index, chunk in enumerate(
            page_chunks
        ):
            chunk_id = create_chunk_id(
                source=pdf_path.name,
                page=page_number,
                chunk_index=chunk_index,
                text=chunk,
            )

            records.append(
                {
                    "id": chunk_id,
                    "document": chunk,
                    "metadata": {
                        "source": (
                            pdf_path.name
                        ),
                        "page": (
                            page_number
                        ),
                        "chunk_index": (
                            chunk_index
                        ),
                        "extraction_type": (
                            extraction_type
                        ),
                    },
                }
            )

    return records
