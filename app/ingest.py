from __future__ import annotations

from typing import Any

import chromadb
import requests

from app.config import (
    CHROMA_BATCH_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_REQUEST_TIMEOUT,
    ensure_directories,
    validate_config,
)
from app.utils import extract_pdf_chunks


def check_ollama() -> None:
    """
    Ollama 서버와 임베딩 모델이 준비되어 있는지 확인한다.
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=OLLAMA_CONNECT_TIMEOUT,
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama 서버에 연결할 수 없습니다.\n"
            "다음 명령으로 상태를 확인하세요:\n"
            "systemctl status ollama\n"
            "또는\n"
            "ollama serve"
        ) from exc

    response_data = response.json()
    models = response_data.get("models", [])

    model_names = {
        model.get("name", "")
        for model in models
    }

    model_exists = any(
        name == EMBEDDING_MODEL
        or name.startswith(f"{EMBEDDING_MODEL}:")
        for name in model_names
    )

    if not model_exists:
        available_models = ", ".join(
            sorted(model_names)
        ) or "없음"

        raise RuntimeError(
            f"임베딩 모델을 찾을 수 없습니다: "
            f"{EMBEDDING_MODEL}\n"
            f"현재 설치 모델: {available_models}\n"
            f"설치 명령:\n"
            f"ollama pull {EMBEDDING_MODEL}"
        )

    print(f"[Ollama 확인] {EMBEDDING_MODEL}")


def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Ollama API를 이용하여 여러 텍스트를 임베딩한다.
    """
    if not texts:
        return []

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": texts,
            },
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        error_detail = ""

        if exc.response is not None:
            error_detail = (
                f"\nOllama 응답: "
                f"{exc.response.text}"
            )

        raise RuntimeError(
            f"임베딩 요청에 실패했습니다: {exc}"
            f"{error_detail}"
        ) from exc

    response_data = response.json()
    embeddings = response_data.get("embeddings")

    if not isinstance(embeddings, list):
        raise RuntimeError(
            "Ollama 응답에 embeddings가 없습니다.\n"
            f"응답: {response_data}"
        )

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "요청한 텍스트 수와 반환된 임베딩 수가 "
            "일치하지 않습니다.\n"
            f"요청 수: {len(texts)}\n"
            f"반환 수: {len(embeddings)}"
        )

    return embeddings


def generate_embeddings(
    records: list[dict[str, Any]],
) -> None:
    """
    모든 청크에 임베딩 값을 추가한다.
    """
    total = len(records)

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE,
    ):
        end = min(
            start + EMBED_BATCH_SIZE,
            total,
        )

        batch = records[start:end]

        texts = [
            record["document"]
            for record in batch
        ]

        embeddings = embed_texts(texts)

        for record, embedding in zip(
            batch,
            embeddings,
        ):
            record["embedding"] = embedding

        print(f"[임베딩 생성] {end}/{total}")


def create_collection(
    client: chromadb.PersistentClient,
) -> chromadb.Collection:
    """
    기존 컬렉션을 삭제하고 새 컬렉션을 생성한다.

    ingest.py를 다시 실행해도 중복 데이터가 생기지 않도록
    기존 컬렉션을 초기화한다.
    """
    try:
        client.delete_collection(
            CHROMA_COLLECTION_NAME
        )

        print(
            "[컬렉션 초기화] "
            f"{CHROMA_COLLECTION_NAME}"
        )

    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "description": (
                "건국대학교 정보화 및 정보보호 관련 규정"
            ),
            "embedding_model": EMBEDDING_MODEL,
            "hnsw:space": "cosine",
        },
    )

    return collection


def save_records(
    collection: chromadb.Collection,
    records: list[dict[str, Any]],
) -> None:
    """
    청크, 메타데이터, 임베딩을 ChromaDB에 저장한다.
    """
    total = len(records)

    for start in range(
        0,
        total,
        CHROMA_BATCH_SIZE,
    ):
        end = min(
            start + CHROMA_BATCH_SIZE,
            total,
        )

        batch = records[start:end]

        collection.add(
            ids=[
                record["id"]
                for record in batch
            ],
            documents=[
                record["document"]
                for record in batch
            ],
            metadatas=[
                record["metadata"]
                for record in batch
            ],
            embeddings=[
                record["embedding"]
                for record in batch
            ],
        )

        print(f"[Chroma 저장] {end}/{total}")


def main() -> None:
    print("=" * 60)
    print("규정 PDF 인덱싱 시작")
    print("=" * 60)

    validate_config()
    ensure_directories()
    check_ollama()

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"PDF 파일이 없습니다: {DOCUMENTS_DIR}"
        )

    print(f"[PDF 개수] {len(pdf_files)}")
    print(f"[Chroma 경로] {CHROMA_DIR}")

    all_records: list[dict[str, Any]] = []

    for pdf_file in pdf_files:
        records = extract_pdf_chunks(pdf_file)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "PDF에서 추출된 청크가 없습니다."
        )

    print(
        f"\n[전체 청크 수] "
        f"{len(all_records)}"
    )

    generate_embeddings(all_records)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = create_collection(client)

    save_records(
        collection=collection,
        records=all_records,
    )

    print("\n" + "=" * 60)
    print("인덱싱 완료")
    print(
        f"컬렉션: "
        f"{CHROMA_COLLECTION_NAME}"
    )
    print(
        f"저장된 청크 수: "
        f"{collection.count()}"
    )
    print(
        f"저장 경로: "
        f"{CHROMA_DIR}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
