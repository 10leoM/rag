# -*- coding: utf-8 -*-
"""F2 · Milvus schema 对齐测试：字段、插入检索、删除。"""

from app.infrastructure.vectordb import (
    CHUNK_INDEX_FIELD,
    DOCUMENT_ID_FIELD,
    EMBEDDING_FIELD,
    FILENAME_FIELD,
    ID_FIELD,
    METADATA_FIELD,
    MIME_TYPE_FIELD,
    TEXT_FIELD,
    MilvusManager,
    MilvusRecord,
)


async def test_create_collection_schema(
    manager: MilvusManager,
    collection: str,
) -> None:
    """创建的集合字段齐全。"""
    from pymilvus import Collection

    col = Collection(collection)
    names = {f.name for f in col.schema.fields}
    expected = {
        ID_FIELD,
        EMBEDDING_FIELD,
        TEXT_FIELD,
        DOCUMENT_ID_FIELD,
        CHUNK_INDEX_FIELD,
        FILENAME_FIELD,
        MIME_TYPE_FIELD,
        METADATA_FIELD,
    }
    assert expected.issubset(names)


async def test_insert_and_search(
    manager: MilvusManager,
    collection: str,
    chunk_record: MilvusRecord,
) -> None:
    """插入后能按主键检索到 text 与元数据。"""
    ids = await manager.insert(collection, [chunk_record])
    assert ids == ["chunk-001"]

    hits = await manager.search(collection, chunk_record.embedding, top_k=1)
    assert len(hits) == 1
    assert hits[0][ID_FIELD] == "chunk-001"
    assert hits[0][TEXT_FIELD] == chunk_record.text
    assert hits[0][DOCUMENT_ID_FIELD] == "doc-001"
    assert hits[0][CHUNK_INDEX_FIELD] == 0


async def test_delete(
    manager: MilvusManager,
    collection: str,
    chunk_record: MilvusRecord,
) -> None:
    """按 id 删除后检索不到。"""
    await manager.insert(collection, [chunk_record])
    await manager.delete(collection, ids=["chunk-001"])

    hits = await manager.search(collection, chunk_record.embedding, top_k=1)
    assert len(hits) == 0
