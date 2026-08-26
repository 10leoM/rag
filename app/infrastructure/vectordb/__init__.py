# -*- coding: utf-8 -*-
"""向量数据库封装（Milvus）。"""

from app.infrastructure.vectordb.milvus_client import (
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

__all__ = [
    "MilvusManager",
    "MilvusRecord",
    "ID_FIELD",
    "EMBEDDING_FIELD",
    "TEXT_FIELD",
    "DOCUMENT_ID_FIELD",
    "CHUNK_INDEX_FIELD",
    "FILENAME_FIELD",
    "MIME_TYPE_FIELD",
    "METADATA_FIELD",
]
