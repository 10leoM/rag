# -*- coding: utf-8 -*-
"""F1 · LocalEmbedder 测试：维度、归一化、相似度、懒加载。"""

import math

import numpy as np

from app.infrastructure.embedding import LocalEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def test_embed_query_dimension_and_normalization(embedder: LocalEmbedder) -> None:
    """单条查询向量维度为 512，且 L2 范数接近 1。"""
    vec = embedder.embed_query("有限责任公司的股东人数上限是多少")
    assert len(vec) == 512
    norm = float(np.linalg.norm(vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-3)


def test_embed_documents_shape(embedder: LocalEmbedder) -> None:
    """批量编码数量与输入一致，每条维度为 512。"""
    texts = ["你好", "世界", "有限责任公司的股东人数上限是多少"]
    vecs = embedder.embed_documents(texts)
    assert len(vecs) == len(texts)
    assert all(len(v) == 512 for v in vecs)


def test_similarity_ordering(
    embedder: LocalEmbedder,
    similarity_pairs: dict,
) -> None:
    """相近句对相似度高，无关句对相似度低。"""
    for a, b in similarity_pairs["similar_pairs"]:
        sim = _cosine(embedder.embed_query(a), embedder.embed_query(b))
        assert sim > 0.7, f"相近句对相似度过低: {a!r} / {b!r} -> {sim:.4f}"
    for a, b in similarity_pairs["dissimilar_pairs"]:
        sim = _cosine(embedder.embed_query(a), embedder.embed_query(b))
        assert sim < 0.4, f"无关句对相似度过高: {a!r} / {b!r} -> {sim:.4f}"


def test_model_not_loaded_on_init() -> None:
    """构造后模型未加载，证明懒加载行为。"""
    fresh = LocalEmbedder(
        model_name="BAAI/bge-small-zh-v1.5",
        device=None,
        expected_dim=512,
    )
    assert fresh._model is None
