from __future__ import annotations

import os
from pathlib import Path


# 프로젝트 최상위 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# 문서와 데이터 저장 경로
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"

# ChromaDB 설정
CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "university_regulations",
)

# Ollama 서버 주소
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

# Ollama 모델
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding:0.6b",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:8b",
)

# PDF 청크 설정
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "900")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)

# 임베딩 요청 배치 크기
EMBED_BATCH_SIZE = int(
    os.getenv("EMBED_BATCH_SIZE", "16")
)

# ChromaDB 저장 배치 크기
CHROMA_BATCH_SIZE = int(
    os.getenv("CHROMA_BATCH_SIZE", "100")
)

# RAG 검색 설정
TOP_K = 5

# cosine distance 기준
# 값이 낮을수록 질문과 문서가 더 유사하다.
MAX_DISTANCE = 0.65

# HTTP 요청 제한 시간
OLLAMA_CONNECT_TIMEOUT = int(
    os.getenv("OLLAMA_CONNECT_TIMEOUT", "10")
)

OLLAMA_REQUEST_TIMEOUT = int(
    os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")
)


def validate_config() -> None:
    """
    설정값이 올바른지 검사한다.
    ingest, rag, api 실행 전에 호출할 수 있다.
    """
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE는 0보다 커야 합니다.")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP은 0 이상이어야 합니다.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다."
        )

    if EMBED_BATCH_SIZE <= 0:
        raise ValueError(
            "EMBED_BATCH_SIZE는 0보다 커야 합니다."
        )

    if CHROMA_BATCH_SIZE <= 0:
        raise ValueError(
            "CHROMA_BATCH_SIZE는 0보다 커야 합니다."
        )

    if TOP_K <= 0:
        raise ValueError("TOP_K는 0보다 커야 합니다.")

    if not 0 <= MAX_DISTANCE <= 2:
        raise ValueError(
            "MAX_DISTANCE는 0 이상 2 이하로 설정하세요."
        )


def ensure_directories() -> None:
    """
    프로젝트에서 필요한 디렉터리를 생성한다.
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
