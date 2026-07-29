from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
import requests

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
    TOP_K,
)


@dataclass
class SearchResult:
    """
    검색 결과 한 개를 표현하는 자료형
    """

    document: str
    source: str
    page: int
    chunk_index: int
    distance: float


def embed_query(query: str) -> list[float]:
    """
    사용자 질문을 Ollama 임베딩 모델로 벡터화한다.
    """
    query = query.strip()

    if not query:
        raise ValueError("질문이 비어 있습니다.")

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": query,
            },
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        detail = ""

        if exc.response is not None:
            detail = f"\nOllama 응답: {exc.response.text}"

        raise RuntimeError(
            f"질문 임베딩 생성에 실패했습니다: {exc}{detail}"
        ) from exc

    data = response.json()
    embeddings = data.get("embeddings")

    if not isinstance(embeddings, list):
        raise RuntimeError(
            f"올바르지 않은 Ollama 응답입니다: {data}"
        )

    if not embeddings:
        raise RuntimeError(
            "Ollama가 임베딩을 반환하지 않았습니다."
        )

    first_embedding = embeddings[0]

    if not isinstance(first_embedding, list):
        raise RuntimeError(
            f"임베딩 형식이 올바르지 않습니다: {data}"
        )

    return first_embedding


def get_collection() -> chromadb.Collection:
    """
    ChromaDB에서 저장된 규정 컬렉션을 불러온다.
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

    except Exception as exc:
        raise RuntimeError(
            "ChromaDB 컬렉션을 찾을 수 없습니다.\n"
            "먼저 다음 명령을 실행하세요:\n"
            "python -m app.ingest"
        ) from exc

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB 컬렉션은 존재하지만 데이터가 없습니다.\n"
            "다음 명령을 다시 실행하세요:\n"
            "python -m app.ingest"
        )

    return collection


def search_regulations(
    query: str,
    top_k: int = TOP_K,
    max_distance: float | None = MAX_DISTANCE,
) -> list[SearchResult]:
    """
    질문과 유사한 규정 청크를 검색한다.

    cosine distance는 값이 낮을수록 질문과 문서가 유사하다.

    max_distance가 None이면 거리 제한 없이 상위 결과를 반환한다.
    """
    if top_k <= 0:
        raise ValueError("top_k는 0보다 커야 합니다.")

    if max_distance is not None and max_distance < 0:
        raise ValueError(
            "max_distance는 0 이상이거나 None이어야 합니다."
        )

    query_embedding = embed_query(query)
    collection = get_collection()

    result_count = min(
        top_k,
        collection.count(),
    )

    raw_result: dict[str, Any] = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents_list = raw_result.get("documents")
    metadatas_list = raw_result.get("metadatas")
    distances_list = raw_result.get("distances")

    if not documents_list:
        return []

    if not metadatas_list:
        return []

    if not distances_list:
        return []

    documents = documents_list[0]
    metadatas = metadatas_list[0]
    distances = distances_list[0]

    search_results: list[SearchResult] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if document is None or metadata is None:
            continue

        numeric_distance = float(distance)

        if (
            max_distance is not None
            and numeric_distance > max_distance
        ):
            continue

        search_results.append(
            SearchResult(
                document=str(document),
                source=str(
                    metadata.get(
                        "source",
                        "알 수 없는 문서",
                    )
                ),
                page=int(
                    metadata.get(
                        "page",
                        0,
                    )
                ),
                chunk_index=int(
                    metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
                distance=numeric_distance,
            )
        )

    return search_results


def print_search_results(
    results: list[SearchResult],
) -> None:
    """
    터미널에서 검색 결과를 확인하기 위한 출력 함수
    """
    if not results:
        print("\n관련 규정을 찾을 수 없습니다.")
        return

    print(
        f"\n검색 결과 {len(results)}개를 찾았습니다."
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(f"[검색 결과 {index}]")
        print(f"출처: {result.source}")
        print(f"페이지: {result.page}")
        print(f"청크 번호: {result.chunk_index}")
        print(f"거리: {result.distance:.4f}")
        print("-" * 70)
        print(result.document)


def main() -> None:
    """
    rag.py 단독 테스트용 함수
    """
    print("=" * 60)
    print("규정 검색 테스트")
    print("=" * 60)

    query = input("질문을 입력하세요: ").strip()

    try:
        results = search_regulations(query)

    except Exception as exc:
        print(f"\n오류: {exc}")
        return

    print_search_results(results)


if __name__ == "__main__":
    main()
