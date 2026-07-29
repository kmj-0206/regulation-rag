from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    ensure_directories,
    validate_config,
)
from app.llm import answer_question
from app.rag import get_collection


class ChatRequest(BaseModel):
    """
    POST /chat 요청 형식
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="규정에 대해 질문할 내용",
        examples=[
            "정보보호위원회는 어떤 사항을 심의하나요?"
        ],
    )


class SourceResponse(BaseModel):
    """
    답변 근거로 사용된 출처
    """

    source: str
    page: int
    chunk_index: int
    distance: float


class ChatResponse(BaseModel):
    """
    POST /chat 응답 형식
    """

    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    """
    GET /health 응답 형식
    """

    status: str
    collection: str
    document_chunks: int
    embedding_model: str
    llm_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버 시작 시 설정과 ChromaDB를 검사한다.
    """
    validate_config()
    ensure_directories()

    try:
        collection = get_collection()
        chunk_count = collection.count()

        print("=" * 60)
        print("규정 RAG API 서버 시작")
        print(f"컬렉션: {CHROMA_COLLECTION_NAME}")
        print(f"저장 청크: {chunk_count}")
        print(f"임베딩 모델: {EMBEDDING_MODEL}")
        print(f"LLM 모델: {LLM_MODEL}")
        print("=" * 60)

    except Exception as exc:
        print(f"[시작 경고] {exc}")

    yield

    print("규정 RAG API 서버 종료")


app = FastAPI(
    title="건국대학교 규정 RAG API",
    description=(
        "건국대학교 정보화 및 정보보호 관련 규정을 "
        "검색하고 답변하는 API"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# 개발 단계 CORS 설정
# React 또는 Spring Boot에서 직접 호출할 때 필요하다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    """
    기본 주소 확인용
    """
    return {
        "message": "건국대학교 규정 RAG API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """
    서버와 ChromaDB 상태 확인
    """
    try:
        collection = get_collection()
        chunk_count = collection.count()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return HealthResponse(
        status="ok",
        collection=CHROMA_COLLECTION_NAME,
        document_chunks=chunk_count,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    사용자 질문을 받아 RAG 기반 답변을 생성한다.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="질문이 비어 있습니다.",
        )

    try:
        result: dict[str, Any] = answer_question(question)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="답변 생성 중 내부 오류가 발생했습니다.",
        ) from exc

    raw_sources = result.get("sources", [])

    sources = [
        SourceResponse(
            source=str(source["source"]),
            page=int(source["page"]),
            chunk_index=int(source["chunk_index"]),
            distance=float(source["distance"]),
        )
        for source in raw_sources
    ]

    return ChatResponse(
        answer=str(result["answer"]),
        sources=sources,
    )
