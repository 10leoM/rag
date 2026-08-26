# -*- coding: utf-8 -*-
"""文档生命周期服务：上传入库、删除、重新分块入库。"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from loguru import logger
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl import ETLPipeline
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.embedding import LocalEmbedder
from app.infrastructure.vectordb import (
    DOCUMENT_ID_FIELD,
    MilvusManager,
    MilvusRecord,
)

UPLOAD_DIR = Path("uploads")


class DocumentService:
    """封装文档上传、删除与重新入库的完整生命周期。"""

    def __init__(
        self,
        embedder: LocalEmbedder,
        milvus: MilvusManager,
        collection_name: str,
    ) -> None:
        self._embedder = embedder
        self._milvus = milvus
        self._collection = collection_name

    async def ensure_collection(self) -> None:
        """确保向量集合存在（幂等，供启动或首次入库调用）。"""
        await self._milvus.create_collection(
            self._collection,
            self._embedder.expected_dim,
        )
        logger.info(
            "document_service.py::ensure_collection 集合已就绪 collection={}",
            self._collection,
        )

    def _build_records(
        self,
        document_id: str,
        filename: str,
        mime_type: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> list[MilvusRecord]:
        """将分块与向量组装为 Milvus 记录。"""
        records: list[MilvusRecord] = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_id = str(uuid.uuid4())
            records.append(
                MilvusRecord(
                    id=chunk_id,
                    embedding=vec,
                    text=chunk,
                    document_id=document_id,
                    chunk_index=i,
                    filename=filename,
                    mime_type=mime_type,
                    metadata={},
                )
            )
        return records

    async def ingest(
        self,
        session: AsyncSession,
        data: bytes,
        filename: str,
        mime_type: str | None,
        document_id: str | None = None,
    ) -> str:
        """上传文档：落盘、解析分块、向量化、入库。"""
        doc_id = document_id or str(uuid.uuid4())

        # 1. 保存原始文件到磁盘
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"{doc_id}_{filename}"
        await asyncio.to_thread(dest.write_bytes, data)

        try:
            # 2. 解析并分块
            pipeline = ETLPipeline()
            etl = await pipeline.run_bytes(data, filename, mime_type)
            chunks = etl.chunks

            # 3. 向量化
            vectors = await self._embedder.aembed_documents(chunks)

            # 4. 组装记录并写入 Milvus
            records = self._build_records(
                doc_id,
                filename,
                mime_type or "",
                chunks,
                vectors,
            )
            await self._milvus.insert(self._collection, records)

            # 5. 写 Postgres，回填 vector_id（= Milvus 主键）
            doc = Document(
                id=doc_id,
                filename=filename,
                mime_type=mime_type,
                storage_path=str(dest),
                status="ready",
                meta={"chunk_count": len(chunks)},
            )
            session.add(doc)
            for rec in records:
                session.add(
                    DocumentChunk(
                        id=rec.id,
                        document_id=doc_id,
                        chunk_index=rec.chunk_index,
                        content=rec.text,
                        vector_id=rec.id,
                        meta=None,
                    )
                )
            await session.commit()
        except Exception:
            await session.rollback()
            await asyncio.to_thread(dest.unlink, True)
            raise

        logger.info(
            "document_service.py::ingest 入库完成 document_id={} chunks={}",
            doc_id,
            len(chunks),
        )
        return doc_id

    async def delete(self, session: AsyncSession, document_id: str) -> None:
        """删除文档：先 Milvus，再 Postgres，最后磁盘。"""
        # 1. 查文档元数据
        doc = await session.get(Document, document_id)
        if doc is None:
            logger.warning(
                "document_service.py::delete 文档不存在 document_id={}",
                document_id,
            )
            return

        # 2. 删 Milvus 向量（按 document_id）
        await self._milvus.delete(
            self._collection,
            expr=f'{DOCUMENT_ID_FIELD} == "{document_id}"',
        )

        # 3. 删 Postgres 分块与文档
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        await session.delete(doc)
        await session.commit()

        # 4. 删磁盘文件
        if doc.storage_path:
            await asyncio.to_thread(Path(doc.storage_path).unlink, True)

        logger.info(
            "document_service.py::delete 删除完成 document_id={}",
            document_id,
        )

    async def reindex(self, session: AsyncSession, document_id: str) -> str:
        """重新分块入库：删旧向量与分块，重读文件再入库。"""
        doc = await session.get(Document, document_id)
        if doc is None:
            raise ValueError(f"文档不存在: {document_id}")
        if not doc.storage_path or not Path(doc.storage_path).exists():
            raise ValueError(f"文档文件不存在: {doc.storage_path}")

        # 1. 读原文件
        data = await asyncio.to_thread(Path(doc.storage_path).read_bytes)

        # 2. 删旧 Milvus 向量
        await self._milvus.delete(
            self._collection,
            expr=f'{DOCUMENT_ID_FIELD} == "{document_id}"',
        )

        # 3. 删旧 Postgres 分块
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        await session.flush()

        # 4. 重新解析分块
        pipeline = ETLPipeline()
        etl = await pipeline.run_bytes(data, doc.filename, doc.mime_type)
        chunks = etl.chunks

        # 5. 重新向量化
        vectors = await self._embedder.aembed_documents(chunks)

        # 6. 重新写入 Milvus 与 Postgres
        records = self._build_records(
            document_id,
            doc.filename,
            doc.mime_type or "",
            chunks,
            vectors,
        )
        await self._milvus.insert(self._collection, records)
        for rec in records:
            session.add(
                DocumentChunk(
                    id=rec.id,
                    document_id=document_id,
                    chunk_index=rec.chunk_index,
                    content=rec.text,
                    vector_id=rec.id,
                    meta=None,
                )
            )
        doc.status = "ready"
        doc.meta = {"chunk_count": len(chunks)}
        await session.commit()

        logger.info(
            "document_service.py::reindex 重新入库完成 document_id={} chunks={}",
            document_id,
            len(chunks),
        )
        return document_id
