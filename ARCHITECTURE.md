# 企业级 AI Agent 项目架构详解

> 面向有后端基础、首次接触 Agent 项目的同学。本文档基于 `app` 目录源码梳理，不涉及运行。

---

## 一、项目大概做了什么

一句话：这是一个**企业级 AI Agent 服务的可运行骨架**，用 FastAPI 暴露 HTTP 接口，把"对话"和"文档入库"两条业务线搭起来，并在 `app/core` 预留了完整的 Agent 能力（ReAct 推理、Plan-and-Execute 规划、反思、RAG 检索增强、记忆系统、工具调用、意图识别）。

> **关键认知**：当前能端到端跑通的只有"最小对话路径"和"文档上传 ETL 路径"。`core` 里那套复杂的 Agent 编排（Orchestrator / ReAct / Planner / Reflection）虽然代码写好了，但**尚未被 HTTP 层调用接线**。README 所说的"可运行最小骨架"正指此意。理解这一点，就不会奇怪"为什么 chat.py 直接调 ModelRouter，而绕过了 AgentOrchestrator"。

技术栈：FastAPI + LangChain/LangGraph + OpenAI 兼容 API + Milvus（向量库）+ Redis（缓存/短期记忆）+ PostgreSQL（关系库/追踪日志）+ Pydantic v2。

---

## 二、文件目录树与功能

```
project-python/
├── app/
│   ├── __init__.py              # 包声明
│   ├── main.py                  # FastAPI 入口：lifespan 初始化异步 DB 引擎，挂载三个路由
│   ├── config.py                # pydantic-settings 配置：从 .env 读取，lru_cache 单例
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py        # 健康检查：/health 存活探针、/health/ready 就绪探针(查DB)
│   │       ├── chat.py          # 对话：/chat 非流式、/chat/stream SSE 流式
│   │       └── document.py      # 文档：/documents/upload 上传+ETL+入库、/documents 列表
│   │
│   ├── core/                    # 与框架无关的领域核心（业务能力）
│   │   ├── agent/               # Agent 编排：orchestrator/react_agent/planner/reflection
│   │   ├── rag/                 # RAG：retriever(多路检索)/reranker(重排)/generator(生成)
│   │   ├── memory/              # 记忆：short_term(Redis)/long_term(向量库)/manager(协调)
│   │   ├── tools/               # 工具：base/registry/router/builtin(calculator/database/search)
│   │   └── intent/              # 意图识别：recognizer(树形规则+置信度+澄清)
│   │
│   ├── infrastructure/          # 对外部系统的封装（基础设施层）
│   │   ├── llm/                 # model_router(多模型路由)/circuit_breaker(熔断)/types
│   │   ├── vectordb/            # milvus_client(集合生命周期+检索)
│   │   ├── cache/               # redis_cache(KV缓存+语义缓存)
│   │   ├── database/            # models(ORM)/session(异步引擎+会话工厂)
│   │   └── trace/               # tracer(进程内 Span 树追踪)
│   │
│   ├── etl/                     # 文档->向量索引的数据流
│   │   ├── parser.py            # 解析：txt/pdf(pypdf)->纯文本
│   │   ├── chunker.py           # 分块：固定/递归/段落三种策略
│   │   └── pipeline.py          # 流水线：解析->分块->可选回调
│   │
│   └── models/                  # API/领域模型（与 ORM 区分）
│       ├── schemas.py           # Pydantic 请求响应模型
│       └── enums.py            # AgentMode/RetrievalMode/MessageRole/TaskStatus
│
├── pyproject.toml / requirements.txt
├── Dockerfile / docker-compose.yml   # 部署：app+postgres+redis+milvus(+etcd+minio)
└── .env.example
```

> **分层原则**：`core` 只定义"能力"和"接口契约（Protocol）"，不依赖具体中间件；`infrastructure` 才是具体实现。这是典型的依赖倒置 / 六边形架构，目的是让 core 可单测、可替换实现。

---

## 三、架构图

### 3.1 设计意图（完整架构）

```
                         ┌─────────────────────────────────────────────┐
  HTTP 请求               │  接入层  app/api/routes                      │
  ──────────────────────►│  health / chat / document                    │
                         └───────────────┬─────────────────────────────┘
                                         │ (设计上应流入编排器)
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │  Agent 编排  core/agent/orchestrator         │
                         │  IntentContext 决策 -> ReAct 或 Plan-Execute │
                         │  + Reflection 反思 + 降级回退                │
                         └──┬──────────┬──────────┬──────────┬─────────┘
            ┌───────────────┘          │          │          │
            ▼                          ▼          ▼          ▼
   ┌─────────────────┐         ┌────────────┐ ┌─────────┐ ┌──────────┐
   │ core/rag        │         │core/memory │ │core/tools│ │core/intent│
   │ retriever->     │◄────────┤ short+long │ │ registry │ │ recognizer│
   │ reranker->      │         │  manager   │ │  router  │ └────┬─────┘
   │ generator       │         └─────┬──────┘ └────┬────┘      │
   └────────┬────────┘               │             │           │
            │ 全部依赖 Protocol 契约，不依赖具体实现              │
            ▼                        ▼             ▼           ▼
   ════════════════════════════════════════════════════════════════
   infrastructure 层（具体实现）
   ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
   │ llm/         │ │ vectordb/    │ │ cache/   │ │ database/    │
   │ model_router │ │ milvus_client│ │redis_cache│ │ session+ORM  │
   │ +circuit_    │ │              │ │ +语义缓存 │ │              │
   │  breaker     │ │              │ │          │ │              │
   └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └──────┬───────┘
          │                │              │              │
          ▼                ▼              ▼              ▼
     OpenAI API         Milvus          Redis        PostgreSQL
```

### 3.2 当前实际接线的最小路径（真正跑通的两条线）

```
对话线:  POST /chat ─► chat.py ─► IntentRecognizer.recognize/clarify
                                   └─► ModelRouter.chat (经 CircuitBreaker)
                                          └─► Tracer 记 span ─► ChatResponse

流式线:  POST /chat/stream ─► 直接 AsyncOpenAI.stream ─► SSE

文档线:  POST /documents/upload ─► 落盘 ─► ETLPipeline(parse+chunk)
                                   └─► Document/DocumentChunk ORM ─► Postgres
```

> 注意：对话线**没有**经过 AgentOrchestrator、ReAct、Memory、Tools、RAG。这些是写好待接线的。

---

## 四、各部分详细剖析

### 4.1 接入层

**`app/main.py`** — 用 FastAPI 的 `lifespan`（异步上下文管理器）在启动时 `init_engine` 建数据库引擎、`configure_session` 绑定会话工厂，存到 `app.state.engine`；关闭时 `engine.dispose()`。`create_app()` 装配三个路由，统一加前缀 `/api/v1`。

**`app/config.py`** — `Settings(BaseSettings)` 把所有配置（app、openai、database、redis、milvus、log）映射成强类型字段，从 `.env` 读取。`get_settings()` 用 `@lru_cache` 做成进程内单例，避免反复读文件。

**`app/api/routes/health.py`** — 两个探针，后端同学很熟的套路：`/health` 只回 `{"status":"ok"}`（存活，K8s liveness）；`/health/ready` 真去 `SELECT 1` 探数据库（就绪，readiness）。

**`app/api/routes/chat.py`** — 当前唯一有"智能"味的端点。非流式路径做三件事：
1. 意图识别 + 低置信度时拼澄清提示进 system message；
2. `ModelRouter.chat` 路由到模型（带熔断）；
3. `Tracer` 记录 trace_id 链路。

流式路径则直接用 `AsyncOpenAI` 的 `stream=True`，按 `data: {json}\n\n` 的 SSE 格式逐块吐。

**`app/api/routes/document.py`** — 上传：落盘到 `uploads/` -> `ETLPipeline.run_bytes` 解析分块 -> 写 `Document` + 多条 `DocumentChunk` 到 Postgres。注意这里只存了文本块，`vector_id=None` 还没写进 Milvus，所以向量入库这步目前是断的。

### 4.2 Agent 编排层（`core/agent`）

这是项目的"大脑"，也是首次接触 Agent 项目最该花时间的地方。

**`orchestrator.py`** — `AgentOrchestrator` 是总指挥。它依赖四个 **Protocol**（接口契约）：`ModelRouter.get_llm(purpose)`、`MemoryManager`、`ToolRegistry`、`Tracer`。`run()` 流程：意图 `IntentContext` 决定模式（`react` / `plan_execute`，意图优先级可覆盖传入 mode）-> 走 ReAct 或 Plan-Execute -> 若 Plan 失败且开了降级开关，回退 ReAct（标 `degraded=True`）-> 可选反思阶段 -> 包成统一 `AgentResponse`。全程用 tracer 打 span 和 event。

> **教学要点**：orchestrator 里定义的 `ModelRouter` **Protocol**（要 `get_llm`）和 infrastructure 里的 `ModelRouter` **具体类**（只有 `chat`）名字一样但接口不同。也就是说 orchestrator 目前**不能直接吃** infrastructure 的 ModelRouter，中间需要适配。`_LLMAdapter` 就是干这个的，但它期望拿到的是"已经选好的 llm 对象"，而不是路由器本身。这是骨架"待接线"的典型缝隙。

**`react_agent.py`** — 经典 ReAct（Reason+Act）循环。它给 LLM 一套严格的中文提示模板，要求输出 `Thought / Action / Action Input / Final Answer`。`_parse_react_step` 用正则从模型文本里抠出这些字段。循环逻辑：
- 每步先取记忆 -> 拼 prompt（含工具列表 + 历史轨迹）-> 调 LLM -> 解析；
- 若解析出 `Final Answer` 就结束并写记忆；
- 若解析出 `Action` 就调 `tools.invoke` 拿 `Observation` 拼回历史；
- 超 `max_steps` 则失败。

这个"让模型自己决定下一步调哪个工具"就是 Agent 区别于普通 LLM 调用的本质。

**`planner.py`** — Plan-and-Execute 模式。`plan()` 让 LLM 把目标拆成 JSON 子任务列表（每个子任务标 `action_type=tool|reasoning`）；`execute()` 顺序执行，tool 类调工具、reasoning 类用 LLM 汇总；失败时 `replan()` 把已有结果 + 错误喂给 LLM 重新规划，最多重试 `max_replan_attempts` 次（`run_with_replan` 封装）。相比 ReAct 的"边想边做"，它是"先全规划再执行"，适合复杂多步任务。

**`reflection.py`** — 答案质量审查员。`reflect()` 让 LLM 输出 JSON：质量分(0-100)、是否完整、是否疑似幻觉、改进建议。`should_retry_or_warn` 据此给出"是否建议重试/告警"的决策。这是给生产环境加的"自我把关"环节。

### 4.3 RAG 子系统（`core/rag`）

**`retriever.py`** — `MultiRetriever` 三种模式：向量检索（Milvus）、关键词检索（自实现的内存 BM25）、混合检索（两路并行 + RRF 融合）。`_BM25Index` 是手写的 Okapi BM25，`_rrf_fuse` 按 `1/(k+rank)` 给多路结果融合排序。这里用 `asyncio.to_thread` 把同步的 pymilvus / 嵌入调用丢到线程池，避免阻塞事件循环——这是后端异步项目对接同步 SDK 的标准手法。

**`reranker.py`** — `Reranker` 用 sentence-transformers 的 CrossEncoder（默认 `ms-marco-MiniLM-L-6-v2`）对 query-doc 对重新打分，懒加载，`predict` 后取 top_k。重排是为了弥补向量检索"召回粗"的问题，精排一遍。

**`generator.py`** — `RAGGenerator` 把检索到的上下文编号成 `[1][2]`，拼进 prompt 让模型"仅依据上下文作答"，再从答案里正则抠 `[n]` 引用映射回 `RetrievalResult`，产出带 `citations` 的 `RAGResponse`。这就是 RAG = 检索 + 重排 + 引用生成 的完整闭环。

### 4.4 记忆系统（`core/memory`）

**`manager.py`** — `MemoryManager` 协调短/长期记忆，`get_context` 同时取短期历史和长期召回，合成 `MemoryContext`；`save` 写入短期。

**`short_term.py`** — Redis 列表存消息（`RPUSH`/`LRANGE`），用 `tiktoken` 估 token。核心是 `_compress_if_needed`：超过窗口条数或 token 阈值时，保留尾部若干条，把前面的喂给 LLM 摘要成一条 `[历史摘要]`，再重写 Redis 列表。这是"滑动窗口 + 自动压缩"，解决对话越来越长爆 token 的问题。

**`long_term.py`** — 向量库（Milvus）存长期记忆。`store` 嵌入后 insert，`recall` 按会话 ID 过滤做语义检索（注意 `expr` 里对单引号转义防注入），`forget` 按主键删除。短期是"最近几轮"，长期是"跨会话的语义召回"。

### 4.5 工具系统（`core/tools`）

**`base.py`** — `BaseTool` 抽象基类：`name/description/parameters` + `schema_parameters()` 导出 OpenAI tools 风格 JSON Schema + 抽象 `execute`。

**`registry.py`** — `ToolRegistry` 注册中心，`get_tools_description` 生成自然语言工具清单塞进 system prompt。

**`router.py`** — `ToolRouter` 按关键词给候选工具打分选子集（可替换成 LLM 路由），避免把所有工具描述都塞给模型。

**`builtin/`** — 三个示例工具：`CalculatorTool`（用 AST 安全求值，禁函数调用）、`DatabaseQueryTool`（只读 SQL，校验 SELECT 且禁写关键字）、`WebSearchTool`（DuckDuckGo html 抓取的降级实现）。

### 4.6 意图识别（`core/intent`）

**`recognizer.py`** — 树形规则引擎：根意图（问答/任务/文档）-> 子意图，靠关键词命中算置信度。低于阈值（0.55）时 `clarify` 生成澄清话术。它是个"可替换点"——企业里可换成分类模型。注意 chat.py 现在就用了这个，它是少数真正接线进 HTTP 的 core 模块。

### 4.7 ETL（`app/etl`）

**`parser.py`** — 按 MIME/后缀选策略：txt 直读、pdf 用 pypdf 抽页文本，未知类型返回空 + warning。

**`chunker.py`** — 三种分块：`FIXED`（固定窗口 + overlap）、`PARAGRAPH`（按空行段合并）、`RECURSIVE`（按分隔符递归细分，深度限制 24 防爆栈）。用 tiktoken 按 token 计长度。

**`pipeline.py`** — `run_bytes` 串起 parse->chunk，支持 `on_chunks` 异步回调（设计上用来触发向量入库，目前 document.py 没传这个回调）。

### 4.8 基础设施层（`infrastructure`）

**`llm/model_router.py`** — 多模型路由：按 `priority` 分组，同优先级内按 `weight` 加权随机（`random()^(1/weight)` 的技巧让权重大者更靠前）。`chat` 按候选顺序尝试，失败自动降级到下一个，全挂才抛错。每个模型配一个 `CircuitBreaker`。

**`llm/circuit_breaker.py`** — 三态熔断器 Closed->Open->Half-Open。失败累计到阈值跳 Open；过恢复窗口进 Half-Open 放行有限试探；试探成功回 Closed，失败回 Open。`asyncio.Lock` 保并发安全。这是保护下游 LLM 服务的标准韧性设计。

**`vectordb/milvus_client.py`** — `MilvusManager` 异步包装同步 pymilvus SDK（`run_in_executor`），管集合创建/插入/检索/删除。集合 schema 是 `id(VARCHAR 主键) + embedding(FLOAT_VECTOR)`，索引 IVF_FLAT+L2。

**`cache/redis_cache.py`** — 普通 KV 缓存 + **语义缓存**：`semantic_get` 把查询编码成向量，在已存条目里算余弦相似度，>= 阈值就命中。无编码器时回退到字符 n-gram 哈希向量和 Jaccard。语义缓存是"问法不同但语义相同就复用答案"的高级缓存。

**`database/session.py`** — `init_engine` 建异步引擎（含 `normalize_async_database_url` 把同步 URL 转成 `+asyncpg`），`configure_session` 绑全局 session 工厂，`get_async_session` 是 FastAPI 依赖注入用的 async generator（路由层负责 commit）。

**`database/models.py`** — ORM：`Conversation`-`Message`（一对多）、`Document`-`DocumentChunk`（一对多）、`TraceLog`。都用 PG 的 UUID 主键、JSON 字段存 meta。

**`trace/tracer.py`** — 进程内 Span 树：`start_trace` 建根 span，`start_child_span` 建子 span（维护 `parent_span_id`），`end_span` 记结果/错误，`get_trace` 返回浅拷贝。`max_traces` 限内存，超了删最旧。注释说"可替换为导出到 OTLP"，即生产可接 OpenTelemetry。

---

## 五、待接线缝隙（Gap 分析）

下表是源码里"写好但没接上"或"接口对不上"的地方，自己实现一遍时优先处理：

| # | 缝隙 | 位置 | 现状 | 影响 |
|---|------|------|------|------|
| 1 | orchestrator 的 `ModelRouter` Protocol 要求 `get_llm(purpose)`；infrastructure 的 `ModelRouter` 只提供 `chat()` | `core/agent/orchestrator.py` vs `infrastructure/llm/model_router.py` | 接口不兼容，名称重名 | orchestrator 无法直接注入 infra 的路由器，需写适配层 |
| 2 | `AgentOrchestrator` 未被任何路由调用 | `api/routes/chat.py` 直接用 `ModelRouter.chat` | 编排器写好但闲置 | ReAct/Plan/Reflection 全链路未上线 |
| 3 | 文档上传分块后未写入 Milvus | `api/routes/document.py` 中 `vector_id=None`，未调 `on_chunks` 回调 | 只入库了文本块 | RAG 检索无向量数据可用 |
| 4 | `MemoryManager` / `ToolRegistry` / `MultiRetriever` 等未注入编排器与路由 | 各 core 模块独立存在 | 单测可用，端到端未串 | 记忆/工具/RAG 在生产路径上未生效 |
| 5 | `Tracer`（infra，Span 树）与 orchestrator 的 `Tracer` Protocol 方法名不同 | `infrastructure/trace/tracer.py` 提供 `start_trace`；orchestrator Protocol 要 `new_trace_id/start_span/end_span/log_event` | 接口不匹配 | chat.py 用的是 infra 的 Tracer；orchestrator 期望另一套 |

> 这些不是 bug，而是骨架刻意留的扩展点。把它们接起来，就是"实现一遍"的主要工作。

---

## 六、LangChain / LangGraph 使用落差

README 和 `pyproject.toml` 把 `langchain`、`langchain-openai`、`langchain-community`、`langgraph` 列为技术栈，但通读 `app/` 源码：

- `core/agent` 的 ReAct、Planner、Reflection 全是**手写**的提示工程 + 正则/JSON 解析，**未 import** langchain 的 Agent/Graph 编排能力；
- `core/rag` 的检索/重排/生成也是手写拼装，未用 langchain 的 RetrievalQA 链；
- 实际真正使用的 LLM 客户端是 `openai.AsyncOpenAI`（在 `model_router.py` 与 `chat.py` 流式分支）。

结论：当前 LangChain/LangGraph 是"声明依赖、未实质编排"的状态。若要真正用上，典型接法是用 LangGraph 的 StateGraph 替换手写的 ReAct 循环、用 langchain 的 Runnable 抽象统一 LLM 调用——但那会改变 core 现有手写结构。

---

## 七、算法内幕附录

### 7.1 BM25（Okapi）

`retriever.py` 的 `_BM25Index` 实现 Okapi BM25 打分。对查询中的每个词项 `t`，文档 `d` 的得分累加：

```
score(d, q) = Σ_t  IDF(t) * ( f(t,d) * (k1+1) ) / ( f(t,d) + k1*(1 - b + b*|d|/avgdl) )
```

- `f(t,d)`：词项在文档中的词频；
- `|d|`：文档长度（token 数），`avgdl`：平均文档长度；
- `k1=1.5`（词频饱和控制）、`b=0.75`（长度归一化强度）；
- IDF 用平滑公式：`IDF(t) = log(1 + (N - df + 0.5) / (df + 0.5))`，避免负值。

每加一个文档就重算一次 IDF（适合小规模内存索引，大规模需离线索引）。

### 7.2 RRF 倒数排名融合

`_rrf_fuse` 合并多路检索结果。对每个候选 id，按它在各路列表中的排名 `rank`（从 0 计）累加：

```
rrf_score(id) = Σ_lists  1 / (k + rank + 1)      # k=60
```

取累加分 top_k。RRF 的优点是不需要分数可比（向量距离与 BM25 分数量纲不同），只看排名，鲁棒且实现简单。

### 7.3 熔断器三态转换

`circuit_breaker.py` 的状态机：

```
                failure_count >= threshold
   CLOSED ─────────────────────────────► OPEN
     ▲                                      │
     │ half_open 试探成功                    │ recovery_timeout 到期
     │                                      ▼
   CLOSED ◄──────────────────── HALF_OPEN
                                      │
                                      │ half_open 试探失败
                                      ▼
                                    OPEN
```

- **CLOSED**：正常放行，每次失败 `_failure_count++`，到 `failure_threshold(默认5)` 跳 OPEN；
- **OPEN**：快速失败（直接抛 `RuntimeError`），直到 `recovery_timeout(60s)` 到期转 HALF_OPEN；
- **HALF_OPEN**：放行最多 `half_open_max(3)` 次试探，成功回 CLOSED 清零，失败回 OPEN。

`asyncio.Lock` 保证并发下状态转换原子。调用走 `call(func, *args)`：状态检查 -> 放行 -> 成功记 success / 失败记 failure。

### 7.4 语义缓存相似度

`redis_cache.py` 的 `semantic_get`：

1. `_encode_query` 把查询编码成向量；有 `semantic_embedder`（SentenceTransformer）时用其 `encode` 并 L2 归一化；否则回退字符 bigram 哈希到 256 维 + 归一化；
2. 遍历 Redis 里存的条目向量，算 `_cosine = dot(a,b) / (|a||b|)`（向量已归一化时即点积）；
3. 若无嵌入向量，回退 `_token_jaccard`（分词集合的交并比）取较大值；
4. 最高相似度 `>= threshold(0.95)` 则命中，直接复用缓存的答案。

`semantic_set` 把查询向量与答案一起存进 Redis（带 TTL），并在一个索引 key 里登记条目 id 供后续扫描。

---

## 附：关键数据流（完整版目标态）

```
用户提问
  │
  ▼
IntentRecognizer ── IntentContext(confidence, slots, preferred_mode, allowed_tools)
  │
  ▼
AgentOrchestrator.run()
  ├─ MemoryManager.get_context(session_id, query)  ─► 短期历史 + 长期召回
  ├─ mode = react | plan_execute
  │     ├─ ReAct: Thought->Action->工具 invoke->Observation 循环
  │     └─ Plan: 子任务分解->顺序执行->失败 replan
  ├─ (可选) ReflectionAgent.reflect() ─► 质量分/幻觉检查
  └─ MemoryManager.save(assistant 回答)
  │
  ▼
AgentResponse(answer, mode_used, steps, reflection, trace_id)
```

RAG 作为检索增强可旁路注入上下文（`MultiRetriever.retrieve` -> `Reranker.rerank` -> `RAGGenerator.generate`），与 Agent 编排解耦，可独立成链或作为工具被 ReAct 调用。
