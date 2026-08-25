# 企业级教学 RAG 开发规划

> 目标：基于现有参考实现底座，落地一个端到端可运行、可验证、可教学的 RAG 服务。
> 第一版聚焦纯 RAG 链路，不接 Agent 编排；第二版再把 RAG 作为工具接入 Agent。

---

## 一、项目目标

### 1.1 一句话目标

在现有参考实现底座上，把 `core/rag` + `etl` + `infrastructure` 里"写好但没接上"的部分打通，产出可运行、可验证、可教学的 RAG 服务。

### 1.2 要解决的断点

1. 文档上传后只存文本块，`vector_id=None`，没有向量入库。
2. `core/rag` 的 `MultiRetriever / Reranker / RAGGenerator` 全部零接线。
3. 项目里没有 embedding 模型接入点（依赖有 sentence-transformers，未封装）。
4. Milvus collection schema（`id + embedding`）与 retriever 期望（`embedding + text + id`）不一致。

### 1.3 目标功能（用户视角）

- 上传 PDF / TXT / Markdown，系统自动解析、分块、向量化、入库。
- 提问，系统检索相关片段、重排、基于检索上下文生成带引用标注的答案。
- 溯源：答案里的 `[1][2]` 对应原文片段，可查分数与原文。
- 教学可视化：看到命中 chunk、分数、重排前后、最终上下文。
- 评估：量化检索召回与答案质量，验证策略变更是否有效。

### 1.4 用户视角例子

管理员上传《中华人民共和国公司法（2023 修订）》PDF，系统分块入库。学生问："有限责任公司的股东人数上限是多少？"

系统返回：

> 有限责任公司由五十个以下股东出资设立。[1]

学生点 `[1]`，看到原文片段 `第二十四条 有限责任公司由五十个以下股东出资设立。`，以及向量分数、重排分数、来源文档与分块序号。教学面板展示这条答案是"检索 + 生成"而非模型凭空编造，证据可见。

---

## 二、已确认决策

| 决策项 | 选择 |
|--------|------|
| 范围边界 | 第一版纯 RAG；第二版接 Agent 编排 |
| embedding | 本地 sentence-transformers |
| 向量库 | 真实 Milvus |
| 教学可视化 | 轻量 Web 页面 |
| 评估体系 | 手写评估 |
| embedding 模型 | `BAAI/bge-small-zh-v1.5`（512 维） |
| reranker 模型 | `BAAI/bge-reranker-base` |
| 生成 LLM | 第一版复用 OpenAI `ModelRouter` |
| 分块参数 | 可配置，纳入评估对比 |
| RAG 端点 | 独立 `/api/v1/rag/query` |
| 前端 | 轻量、中文友好、无构建步骤 |
| 评估数据集 | 法律 |
| 文档管理 | 上传 + 查询 + 删除 + 重新分块入库 |

---

## 三、功能拆解（里程碑）

### M0 · 基础设施就绪

- **F1** · Embedding 服务封装
- **F2** · Milvus collection schema 对齐
- **F3** · 文档入库与生命周期（上传 / 删除 / 重新分块入库）

### M1 · RAG 核心链路

- **F4** · 检索服务接线（MultiRetriever 三种模式）
- **F5** · 重排 + 生成接线
- **F6** · RAG 查询 API（独立端点）

### M2 · 教学可视化

- **F7** · 检索过程结构化 trace
- **F8** · Web 页面（上传、提问、pipeline 展示）

### M3 · 评估体系

- **F9** · 评估数据集
- **F10** · 检索评估
- **F11** · 答案质量评估
- **F12** · 评估报告与回归

---

## 四、各功能点实现与测试

### M0 · 基础设施

#### F1 · Embedding 封装

实现：
- 新增 `app/infrastructure/embedding/`，核心类 `LocalEmbedder`。
- 接口 `embed_query(text) -> list[float]`、`embed_documents(texts) -> list[list[float]]`，内部 L2 归一化。
- 模型懒加载（首次调用实例化），线程池执行避免阻塞事件循环。
- `config.py` 增字段 `embedding_model_name`、`embedding_dim=512`、`embedding_device`。

测试：
- 数据：8 组中文句对，4 组语义相近，4 组无关。
- 断言：输出维度 == 512；`||v||` 接近 1；相近对余弦相似度 > 0.7，无关对 < 0.4；无网络时给出可读错误。

#### F2 · Milvus schema 对齐

实现：
- `milvus_client.py` 字段扩为 `id(VARCHAR 64 主键)`、`embedding(FLOAT_VECTOR 512)`、`text(VARCHAR 8192)`、`document_id(VARCHAR)`、`chunk_index(INT64)`、`filename(VARCHAR)`、`mime_type(VARCHAR)`、`metadata(JSON)`。
- 索引 IVF_FLAT + L2（nlist 128），补 load/release/flush，`insert/search/delete` 匹配新 schema。

测试：
- 数据：一条带全字段的已知分块。
- 断言：`describe` 字段齐全；按 `id` 查回 text 与元数据；`search` 返回 `id + text + distance`。

#### F3 · 文档入库与生命周期

实现：
- 上传：解析分块后，在 `on_chunks` 回调做 `embed_documents(chunks)` → Milvus 批量 insert → 回填 `DocumentChunk.vector_id`。
- 删除：删 Milvus 向量（按 `document_id`）→ 删 Postgres `Document/DocumentChunk` → 删磁盘文件。清理顺序保证失败时数据一致。
- 重新分块入库：删旧向量 → 重新解析分块 → 重新 embedding + insert → 更新 `vector_id`。
- `LocalEmbedder` + `MilvusManager` 作为应用级单例，在 lifespan 初始化并注入路由。

测试：
- 上传：已知文档，Milvus 向量数 == 分块数，`vector_id` 全非空。
- 删除：Milvus 按 `document_id` 查无向量，Postgres 无记录，磁盘无文件。
- 重新入库：更新后向量数 == 新分块数，旧 `vector_id` 被清理。

### M1 · RAG 核心

#### F4 · 检索接线

实现：
- 新增 `app/services/rag_service.py`，实例化 `MultiRetriever(embedder, milvus_collection)`。
- 从 Postgres/Milvus 拉全量 `id -> text` 注册 BM25，暴露 `retrieve(query, mode, top_k)` 支持 vector / keyword / hybrid。

测试：
- 数据：20 个 chunk 的中文法律文档集，标注 ground-truth chunk。
- 断言：ground-truth chunk 落入 top-k；hybrid 的 recall@5 不低于 vector 与 keyword。

#### F5 · 重排 + 生成

实现：
- `Reranker` 模型换成 `BAAI/bge-reranker-base`（中文，替换默认英文 ms-marco）。
- 链：`retrieve` → `Reranker.rerank` → `RAGGenerator.generate`，返回带引用的 `RAGResponse`。

测试：
- 数据：F4 检索结果 + 标准答案。
- 断言：rerank 后 recall@5 提升或持平；答案含 `[n]`，每个引用映射回上下文，无越界引用。

#### F6 · RAG 查询 API

实现：
- 新增 `POST /api/v1/rag/query`，入参 `query / mode / top_k / rerank_top_k / include_trace`，返回 `RAGResponse`（扩展 `trace` 字段）。

测试：
- 数据：固定 query 的预期答案与 ground-truth chunk。
- 断言：curl 返回结构化 JSON，含 `answer/citations/raw_contexts`，分数字段非空，`include_trace=true` 含中间态。

### M2 · 教学可视化

#### F7 · 检索 trace

实现：
- trace 结构：`query、mode、retrieved[{chunk,score,rank,source}]、reranked[{chunk,score_before,score_after}]、final_contexts、answer、citations`。
- 在 `rag_service` 各阶段收集，透传到 API。

测试：
- 数据：一次已知查询。
- 断言：trace 各阶段字段齐全、顺序一致、rerank 后按分数降序。

#### F8 · Web 页面

实现：
- FastAPI 挂 `StaticFiles` + 单页 HTML，原生 JS + fetch，手写 CSS，无构建步骤。
- 页面：上传文档、提问、逐步展示 pipeline；全中文 UI，UTF-8 + 中文字体栈。

测试：
- Playwright 截图 + 交互断言：上传文档、发起查询、正确渲染命中/重排/引用；无控制台报错、无文字溢出。

### M3 · 评估体系

#### F9 · 评估数据集

实现：
- 一个中文法律文档集 + 手工 QA 集，每条含 `question / reference_answer / ground_truth_chunk_ids`。
- 结构化 JSON 管理，带加载与校验脚本。

测试：
- 断言：可加载、QA 数量 ≥ 100 条、字段完整、`ground_truth_chunk_ids` 真实存在。

#### F10 · 检索评估

实现：
- 指标：`recall@k`、命中率、MRR。
- 评估矩阵：跨 `mode ∈ {vector, keyword, hybrid}`、`chunk_size ∈ {256, 512, 1024}`、`overlap ∈ {0, 64, 128}` 组合跑。

测试：
- 断言：同一查询集下输出可对比表格；参数变化引起的指标差异可量化展示。

#### F11 · 答案质量评估

实现：
- 手写忠实度：答案关键陈述能否在 `final_contexts` 找到原文依据。
- 引用覆盖度：被引用的上下文是否覆盖答案要点。

测试：
- 数据：F9 的 QA 集 + 生成结果。
- 断言：产出 0-1 忠实度分与引用覆盖分；能列出低分案例用于复盘。

#### F12 · 评估报告与回归

实现：
- 一键跑：载入数据 → 检索评估 → 答案质量评估 → 输出 Markdown/HTML 报告（指标表 + 差案例）。
- 可重复执行，参数/策略变更后重跑对比。

测试：
- 断言：单命令产出报告文件；两次运行结果可复现（固定随机种子与模型版本）。

---

## 五、技术选型清单

| 能力 | 选型 |
|------|------|
| embedding | `sentence-transformers` + `BAAI/bge-small-zh-v1.5`（512 维） |
| 重排 | `BAAI/bge-reranker-base`（CrossEncoder） |
| 生成 LLM | OpenAI 兼容 API（复用 `ModelRouter`） |
| 向量库 | Milvus（IVF_FLAT + L2） |
| 关系库 | PostgreSQL（文档与分块元数据） |
| 分块 | `DocumentChunker`，可配置 chunk_size / overlap |
| Web | FastAPI StaticFiles + 原生 JS + 手写 CSS |
| 评估 | 手写 recall@k / 命中率 / MRR / 忠实度 / 引用覆盖 |

---

## 六、后续（第二版 A2）

RAG 稳定后，把检索封装成一个 `BaseTool`（如 `RAGQueryTool`），注册进 `ToolRegistry`，让 `AgentOrchestrator` 的 ReAct / Planner 能按需调用 RAG。此版本不在第一版展开。
