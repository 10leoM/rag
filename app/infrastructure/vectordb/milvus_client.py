# -*- coding: utf-8 -*-
"""Milvus 向量库管理器：集合生命周期与向量检索封装。

schema 面向 RAG 分块设计，字段常量集中在模块顶部，供检索层复用。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

from loguru import logger

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
except ImportError:  # pragma: no cover
    Collection = Any  # type: ignore[misc, assignment]
    CollectionSchema = Any  # type: ignore[misc, assignment]
    DataType = Any  # type: ignore[misc, assignment]
    FieldSchema = Any  # type: ignore[misc, assignment]
    connections = Any  # type: ignore[misc, assignment]
    utility = Any  # type: ignore[misc, assignment]


# 1. 字段名常量：schema 与检索层共用，避免魔法字符串
ID_FIELD = "id"
EMBEDDING_FIELD = "embedding"
TEXT_FIELD = "text"
DOCUMENT_ID_FIELD = "document_id"
CHUNK_INDEX_FIELD = "chunk_index"
FILENAME_FIELD = "filename"
MIME_TYPE_FIELD = "mime_type"
METADATA_FIELD = "metadata"

# 2. VARCHAR 字段长度上限
ID_MAX_LENGTH = 64
DOCUMENT_ID_MAX_LENGTH = 64
TEXT_MAX_LENGTH = 8192
FILENAME_MAX_LENGTH = 1024
MIME_TYPE_MAX_LENGTH = 256

# 3. 向量索引参数
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NLIST = 128


def _run_sync(func, *args, **kwargs):
    """在线程池中执行同步 Milvus SDK 调用，避免阻塞事件循环。"""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


@dataclass
class MilvusRecord:
    """一条 RAG 分块记录，字段与 collection schema 一一对应。"""

    id: str
    embedding: list[float]
    text: str
    document_id: str
    chunk_index: int
    filename: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def build_chunk_fields(dim: int) -> list:
    """构建 RAG 分块集合的字段列表。"""
    return [
        FieldSchema(
            name=ID_FIELD,
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=ID_MAX_LENGTH,
        ),
        FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(
            name=TEXT_FIELD,
            dtype=DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
        ),
        FieldSchema(
            name=DOCUMENT_ID_FIELD,
            dtype=DataType.VARCHAR,
            max_length=DOCUMENT_ID_MAX_LENGTH,
        ),
        FieldSchema(name=CHUNK_INDEX_FIELD, dtype=DataType.INT64),
        FieldSchema(
            name=FILENAME_FIELD,
            dtype=DataType.VARCHAR,
            max_length=FILENAME_MAX_LENGTH,
        ),
        FieldSchema(
            name=MIME_TYPE_FIELD,
            dtype=DataType.VARCHAR,
            max_length=MIME_TYPE_MAX_LENGTH,
        ),
        FieldSchema(name=METADATA_FIELD, dtype=DataType.JSON),
    ]


class MilvusManager:
    """Milvus 向量数据库管理器（异步包装同步 SDK）。"""

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        alias: str = "default",
        **conn_kwargs: Any,
    ) -> None:
        self._host = host
        self._port = port
        self._alias = alias
        self._conn_kwargs = conn_kwargs
        self._connected = False

    async def _ensure_connection(self) -> None:
        if self._connected:
            return
        try:

            def _connect() -> None:
                connections.connect(
                    alias=self._alias,
                    host=self._host,
                    port=self._port,
                    **self._conn_kwargs,
                )

            await _run_sync(_connect)
            self._connected = True
            logger.info(
                "milvus_client.py::_ensure_connection 已连接 Milvus {}:{}",
                self._host,
                self._port,
            )
        except Exception as exc:
            logger.exception("milvus_client.py::_ensure_connection 连接失败: {}", exc)
            raise

    async def create_collection(self, name: str, dim: int) -> None:
        """创建 RAG 分块集合；已存在则跳过。"""
        await self._ensure_connection()

        def _create() -> None:
            if utility.has_collection(name, using=self._alias):
                logger.info(
                    "milvus_client.py::create_collection 集合 [{}] 已存在，跳过创建",
                    name,
                )
                return
            # 1. 构建字段与 schema
            fields = build_chunk_fields(dim)
            schema = CollectionSchema(
                fields=fields,
                description=f"RAG 分块集合 {name}",
            )
            # 2. 创建集合
            col = Collection(name, schema, using=self._alias)
            # 3. 建向量索引
            idx = {
                "index_type": INDEX_TYPE,
                "metric_type": METRIC_TYPE,
                "params": {"nlist": NLIST},
            }
            col.create_index(field_name=EMBEDDING_FIELD, index_params=idx)
            logger.info(
                "milvus_client.py::create_collection 已创建集合 [{}] dim={}",
                name,
                dim,
            )

        try:
            await _run_sync(_create)
        except Exception as exc:
            logger.exception(
                "milvus_client.py::create_collection 创建失败 name={}: {}",
                name,
                exc,
            )
            raise

    async def insert(
        self,
        collection: str,
        records: Sequence[MilvusRecord],
    ) -> list[str]:
        """批量插入分块记录，返回主键 id 列表。"""
        await self._ensure_connection()
        if not records:
            return []

        def _insert() -> None:
            col = Collection(collection, using=self._alias)
            # 1. 按 schema 字段顺序组装列数据
            data = [
                [r.id for r in records],
                [r.embedding for r in records],
                [r.text for r in records],
                [r.document_id for r in records],
                [r.chunk_index for r in records],
                [r.filename for r in records],
                [r.mime_type for r in records],
                [r.metadata or {} for r in records],
            ]
            # 2. 插入并落盘
            col.insert(data)
            col.flush()

        try:
            await _run_sync(_insert)
            ids = [r.id for r in records]
            logger.info(
                "milvus_client.py::insert 插入完成 collection={} count={}",
                collection,
                len(ids),
            )
            return ids
        except Exception as exc:
            logger.exception(
                "milvus_client.py::insert 插入失败 collection={}: {}",
                collection,
                exc,
            )
            raise

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        output_fields: list[str] | None = None,
        **search_params: Any,
    ) -> list[dict[str, Any]]:
        """向量检索，返回全字段命中列表。"""
        await self._ensure_connection()
        if output_fields is None:
            output_fields = [
                ID_FIELD,
                TEXT_FIELD,
                DOCUMENT_ID_FIELD,
                CHUNK_INDEX_FIELD,
                FILENAME_FIELD,
                MIME_TYPE_FIELD,
                METADATA_FIELD,
            ]
        params = search_params.get(
            "params",
            {"metric_type": METRIC_TYPE, "params": {"nprobe": 10}},
        )

        def _search() -> list[dict[str, Any]]:
            col = Collection(collection, using=self._alias)
            # 1. 加载集合到内存
            col.load()
            # 2. 向量检索
            res = col.search(
                data=[query_vector],
                anns_field=EMBEDDING_FIELD,
                param=params,
                limit=top_k,
                output_fields=output_fields,
            )
            out: list[dict[str, Any]] = []
            for hits in res:
                for hit in hits:
                    entity = getattr(hit, "entity", None) or {}
                    if hasattr(entity, "to_dict"):
                        entity = entity.to_dict()
                    row: dict[str, Any] = {
                        ID_FIELD: entity.get(ID_FIELD) or hit.id,
                        "distance": float(hit.distance),
                    }
                    for f in output_fields:
                        if f != ID_FIELD:
                            row[f] = entity.get(f)
                    out.append(row)
            return out

        try:
            result = await _run_sync(_search)
            logger.info(
                "milvus_client.py::search 检索完成 collection={} hits={}",
                collection,
                len(result),
            )
            return result
        except Exception as exc:
            logger.exception(
                "milvus_client.py::search 检索失败 collection={}: {}",
                collection,
                exc,
            )
            raise

    async def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        expr: str | None = None,
    ) -> None:
        """按 id 列表或布尔表达式删除。"""
        await self._ensure_connection()
        if not expr and ids:
            expr = f"{ID_FIELD} in [" + ", ".join(f'"{i}"' for i in ids) + "]"
        if not expr:
            return

        def _delete() -> None:
            col = Collection(collection, using=self._alias)
            col.delete(expr)
            col.flush()

        try:
            await _run_sync(_delete)
            logger.info(
                "milvus_client.py::delete 删除完成 collection={} expr={}",
                collection,
                expr,
            )
        except Exception as exc:
            logger.exception(
                "milvus_client.py::delete 删除失败 collection={}: {}",
                collection,
                exc,
            )
            raise

    async def drop_collection(self, name: str) -> None:
        """删除集合。"""
        await self._ensure_connection()

        def _drop() -> None:
            if utility.has_collection(name, using=self._alias):
                utility.drop_collection(name, using=self._alias)

        try:
            await _run_sync(_drop)
            logger.info("milvus_client.py::drop_collection 已删除集合 [{}]", name)
        except Exception as exc:
            logger.exception(
                "milvus_client.py::drop_collection 删除失败 name={}: {}",
                name,
                exc,
            )
            raise
