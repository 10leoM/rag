# -*- coding: utf-8 -*-
"""本地 embedding 服务：封装 sentence-transformers，懒加载 + L2 归一化。"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

import numpy as np
from loguru import logger


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """对向量做 L2 归一化，零向量原样返回。"""
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        return vector / norm
    return vector


class LocalEmbedder:
    """基于 sentence-transformers 的本地嵌入器。

    满足 ``app.core.rag.retriever.EmbeddingProtocol`` 的 ``embed_query`` 约定。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str | None = None,
        expected_dim: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.expected_dim = expected_dim
        self._model: Any = None  # 懒加载：首次调用才实例化
        self._dim_warned = False

    def _load_model(self) -> Any:
        """按需加载模型，优先离线缓存，避免应用启动时阻塞。"""
        if self._model is not None:
            return self._model

        # 1. 延迟导入，未安装时不影响其他模块
        from sentence_transformers import SentenceTransformer

        logger.info(
            "embedder.py::_load_model 开始加载模型 model={} device={}",
            self.model_name,
            self.device or "auto",
        )
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        # 2. 优先离线加载，缓存已存在时无需联网
        try:
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=True,
                **kwargs,
            )
            logger.info(
                "embedder.py::_load_model 离线加载完成 model={}",
                self.model_name,
            )
        except Exception as offline_err:
            # 3. 缓存不存在时回退联网下载
            logger.warning(
                "embedder.py::_load_model 离线加载失败，回退联网: {}",
                offline_err,
            )
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=False,
                **kwargs,
            )
            logger.info(
                "embedder.py::_load_model 联网加载完成 model={}",
                self.model_name,
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[np.ndarray]:
        """编码并做 L2 归一化。"""
        if not texts:
            return []
        model = self._load_model()
        # 1. 批量编码（关闭内部归一化，自行处理以保持维度可验证）
        embeddings = model.encode(
            list(texts),
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        # 2. 逐条 L2 归一化
        vectors = [
            _l2_normalize(np.asarray(v, dtype=np.float32)) for v in embeddings
        ]
        # 3. 校验维度与配置一致（仅提示一次）
        if (
            vectors
            and not self._dim_warned
            and len(vectors[0]) != self.expected_dim
        ):
            logger.warning(
                "embedder.py::_encode 输出维度与配置不符 expected={} actual={}",
                self.expected_dim,
                len(vectors[0]),
            )
            self._dim_warned = True
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询，满足 EmbeddingProtocol。"""
        # 1. 编码取首条
        vector = self._encode([text])[0]
        logger.debug("embedder.py::embed_query 编码完成 dim={}", len(vector))
        # 2. 转成 Python list 返回
        return vector.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档。"""
        vectors = self._encode(texts)
        logger.info(
            "embedder.py::embed_documents 批量编码完成 count={} dim={}",
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
        return [v.tolist() for v in vectors]

    async def aembed_query(self, text: str) -> list[float]:
        """异步编码单条查询，线程池执行避免阻塞事件循环。"""
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """异步批量编码。"""
        return await asyncio.to_thread(self.embed_documents, texts)
