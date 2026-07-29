from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.pool

from app.config import DATABASE_URL, EMBEDDING_DIM


_pool: psycopg2.pool.SimpleConnectionPool | None = None


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    """
    Postgres 연결 풀을 지연 생성하여 반환한다.
    """
    global _pool

    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
        )

    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    """
    연결 풀에서 연결을 하나 빌려 사용하고 반환한다.

    정상 종료 시 commit, 예외 발생 시 rollback 한다.
    """
    pool = _get_pool()
    conn = pool.getconn()

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        pool.putconn(conn)


# 스키마 정의
# - regulation_chunks: 규정 PDF 청크
# - faq_chunks: CommuteMate FAQ 청크 (faq_id는 CommuteMate faq.id 참조)
# - HNSW 인덱스: 코사인 거리 벡터 검색용
# - GIN 인덱스: to_tsvector('simple') 키워드 검색용
SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS regulation_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    page INT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_regulation_chunks_embedding
    ON regulation_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_regulation_chunks_fts
    ON regulation_chunks
    USING gin (to_tsvector('simple', content));

CREATE TABLE IF NOT EXISTS faq_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    faq_id BIGINT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    category_ids BIGINT[] NOT NULL DEFAULT '{{}}',
    faq_created_at DATE,
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_faq_chunks_faq_id
    ON faq_chunks (faq_id);

CREATE INDEX IF NOT EXISTS idx_faq_chunks_embedding
    ON faq_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_faq_chunks_fts
    ON faq_chunks
    USING gin (to_tsvector('simple', content));
"""


def to_vector_literal(
    embedding: list[float],
) -> str:
    """
    임베딩 리스트를 pgvector 리터럴 문자열로 변환한다.

    SQL에서 %s::vector 형태로 바인딩하여 사용한다.
    """
    return "[" + ",".join(
        repr(float(value))
        for value in embedding
    ) + "]"


def init_schema() -> None:
    """
    pgvector 확장, 청크 테이블, 인덱스를 생성한다.

    이미 존재하면 아무 것도 하지 않는다 (IF NOT EXISTS).
    서버 시작(lifespan)과 ingest 실행 시 호출된다.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def check_db() -> dict[str, int]:
    """
    DB 연결과 청크 테이블 상태를 확인한다.

    반환: 테이블별 청크 수
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM regulation_chunks"
            )
            regulation_count = cur.fetchone()[0]

            cur.execute(
                "SELECT count(*) FROM faq_chunks"
            )
            faq_count = cur.fetchone()[0]

    return {
        "regulation_chunks": regulation_count,
        "faq_chunks": faq_count,
    }
