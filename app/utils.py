from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def normalize_text(text: str) -> str:
    """
    PDF에서 추출된 텍스트의 불필요한 공백을 정리한다.

    - 연속된 공백 제거
    - 빈 줄 제거
    - 줄 단위 구조는 최대한 유지
    """
    normalized_lines: list[str] = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.split())

        if cleaned_line:
            normalized_lines.append(cleaned_line)

    return "\n".join(normalized_lines)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    긴 텍스트를 여러 청크로 분리한다.

    가능하면 줄바꿈이나 문장 종료 지점에서 자른다.
    청크 사이에는 overlap만큼 내용을 겹쳐 포함한다.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")

    if overlap < 0:
        raise ValueError("overlap은 0 이상이어야 합니다.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap은 chunk_size보다 작아야 합니다."
        )

    chunks: list[str] = []
    text_length = len(text)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            search_start = start + int(chunk_size * 0.6)

            split_candidates = [
                text.rfind("\n", search_start, end),
                text.rfind(". ", search_start, end),
                text.rfind("다. ", search_start, end),
                text.rfind("조 ", search_start, end),
            ]

            split_position = max(split_candidates)

            if split_position > start:
                end = split_position + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

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
    """
    청크마다 고유한 ID를 생성한다.

    동일한 파일, 페이지, 청크 내용이면 항상 동일한 ID가 생성된다.
    """
    raw_value = (
        f"{source}|{page}|{chunk_index}|{text}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


def extract_pdf_chunks(
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """
    PDF 한 개를 읽어서 페이지별 청크 목록으로 변환한다.

    반환 형식:
    [
        {
            "id": "...",
            "document": "...",
            "metadata": {
                "source": "...pdf",
                "page": 1,
                "chunk_index": 0
            }
        }
    ]
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"PDF 파일이 아닙니다: {pdf_path}"
        )

    print(f"\n[PDF 읽기] {pdf_path.name}")

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    records: list[dict[str, Any]] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        raw_text = page.extract_text() or ""
        normalized_text = normalize_text(raw_text)

        if not normalized_text:
            print(
                f"  - {page_number}페이지: "
                "추출된 텍스트 없음"
            )
            continue

        page_chunks = split_text(normalized_text)

        print(
            f"  - {page_number}페이지: "
            f"{len(normalized_text):,}자, "
            f"{len(page_chunks)}개 청크"
        )

        for chunk_index, chunk in enumerate(page_chunks):
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
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return records
