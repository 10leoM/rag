# -*- coding: utf-8 -*-
"""F3 · DocumentService 测试：上传、删除、重新入库。"""

from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select

from app.infrastructure.database.models import Document, DocumentChunk
from app.services import DocumentService

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "services"


def _read_document() -> bytes:
    path = DATA_DIR / "document.txt"
    return path.read_bytes()


async def _chunk_count(session, document_id: str) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    )
    return int(result.scalar() or 0)


def _query_count(collection: str, document_id: str) -> int:
    """按 document_id 过滤查询存活实体数（软删除的不可见）。"""
    from pymilvus import Collection

    col = Collection(collection)
    col.load()
    res = col.query(
        expr=f'document_id == "{document_id}"',
        output_fields=["id"],
    )
    data = getattr(res, "data", res)
    return len(data)


async def test_ingest(
    service: DocumentService,
    db_session,
    collection: str,
) -> None:
    """上传后向量数等于分块数，vector_id 全非空。"""
    doc_id = str(uuid4())
    await service.ingest(
        db_session,
        _read_document(),
        "document.txt",
        "text/plain",
        document_id=doc_id,
    )

    chunk_count = await _chunk_count(db_session, doc_id)
    assert chunk_count > 0

    vectors = await db_session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    )
    chunks = vectors.scalars().all()
    assert all(c.vector_id for c in chunks)
    assert len(chunks) == chunk_count
    assert _query_count(collection, doc_id) == chunk_count

    await service.delete(db_session, doc_id)


async def test_delete(
    service: DocumentService,
    db_session,
    collection: str,
) -> None:
    """删除后 Milvus、Postgres、磁盘均无残留。"""
    doc_id = str(uuid4())
    await service.ingest(
        db_session,
        _read_document(),
        "document.txt",
        "text/plain",
        document_id=doc_id,
    )

    doc = await db_session.get(Document, doc_id)
    storage_path = doc.storage_path

    await service.delete(db_session, doc_id)

    assert _query_count(collection, doc_id) == 0
    assert await db_session.get(Document, doc_id) is None
    assert await _chunk_count(db_session, doc_id) == 0
    assert not Path(storage_path).exists()


async def test_reindex(
    service: DocumentService,
    db_session,
    collection: str,
) -> None:
    """重新入库后向量数等于新分块数，旧向量被清理。"""
    doc_id = str(uuid4())
    await service.ingest(
        db_session,
        _read_document(),
        "document.txt",
        "text/plain",
        document_id=doc_id,
    )
    old_ids = {
        c.id
        for c in (
            await db_session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
            )
        )
        .scalars()
        .all()
    }

    await service.reindex(db_session, doc_id)

    new_chunks = (
        await db_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        )
    ).scalars().all()
    new_ids = {c.id for c in new_chunks}

    assert new_chunks
    assert new_ids != old_ids
    assert _query_count(collection, doc_id) == len(new_chunks)

    await service.delete(db_session, doc_id)
