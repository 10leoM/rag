# 验收命令

> 每完成一个功能点，在此追加对应验收命令。改动后重跑，出现预期输出即验收通过。

## M0 · 基础设施

### M0-F1 · Embedding 封装

```powershell
# 在项目根目录执行，首次运行会下载 bge-small-zh-v1.5 模型
.venv\Scripts\python.exe -m pytest tests/infrastructure/embedding/test_embedder.py -v
```

期望输出：`4 passed`。

### M0-F2 · Milvus schema 对齐

```powershell
# 前置：Docker Desktop 已启动，Milvus 容器运行中
docker compose up -d milvus
# 在项目根目录执行
.venv\Scripts\python.exe -m pytest tests/infrastructure/vectordb/test_milvus_client.py -v
```

期望输出：`3 passed`（伴随 PyMilvusDeprecationWarning，属预期，不影响验收）。

### M0-F3 · 文档入库与生命周期

```powershell
# 前置：Docker Desktop 已启动，Milvus 与 Postgres 容器运行中
docker compose up -d milvus postgres
# 在项目根目录执行
.venv\Scripts\python.exe -m pytest tests/services/test_document_service.py -v
```

期望输出：`3 passed`（伴随 PyMilvusDeprecationWarning，属预期，不影响验收）。

## M1 · RAG 核心链路

### M1-F4 · 检索服务接线

```powershell
# 前置：Docker Desktop 已启动，Milvus 容器运行中
docker compose up -d milvus
# 在项目根目录执行
.venv\Scripts\python.exe -m pytest tests/services/test_rag_service.py -v
```

期望输出：`1 passed`（伴随 PyMilvusDeprecationWarning，属预期，不影响验收）。

### M1-F5 · 重排 + 生成接线

```powershell
# 前置：Docker Desktop 已启动，Milvus 容器运行中
docker compose up -d milvus
# 在项目根目录执行，首次运行会下载 bge-reranker-base 模型
.venv\Scripts\python.exe -m pytest tests/services/test_rag_generation.py -v
```

期望输出：`2 passed`（伴随 PyMilvusDeprecationWarning，属预期，不影响验收）。

### M1-F6 · RAG 查询 API

```powershell
# 在项目根目录执行，测试用 Fake 服务注入，不依赖 OpenAI 与 Milvus
.venv\Scripts\python.exe -m pytest tests/api/test_rag_api.py -v
```

期望输出：`2 passed`。
