# 企业级 AI Agent 服务 — 运行指南

本文档描述在本机从零启动该项目所需的所有步骤。按顺序执行即可成功运行。

## 前置环境

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.11+ | 本项目使用 Python 3.12.13 验证通过 |
| PostgreSQL | 16+ | 对话/文档持久化（可选，见下文） |
| Redis | 7+ | 缓存与语义缓存（可选，见下文） |
| Milvus | 2.4+ | 向量检索（可选，见下文） |
| Docker | 24+ | 一键拉起中间件（可选） |

## 一、安装 Python 依赖

```powershell
cd D:\code\ai-agent\project-python

# 创建虚拟环境（已创建则跳过）
C:\Users\25496\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv

# 激活
.\.venv\Scripts\Activate.ps1

# 安装核心依赖（不含 unstructured，见下方说明）
pip install -r requirements-core.txt

# 以可编辑模式安装本项目
pip install -e .
```

注：unstructured 包因其子依赖 langdetect 需要源码编译，在部分环境下会失败。
本项目的文档解析器实际使用 pypdf（已安装），不需要 unstructured 即可运行。
如确需 unstructured，可尝试：pip install --no-cache-dir unstructured。

## 二、配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 .env，至少修改以下项：

```
# 必填：填入你的 OpenAI API Key（或兼容 API 的 Key）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
# 如使用兼容 API（如自建网关），修改 Base URL
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

其余项（数据库、Redis、Milvus）保持默认即可在「最小模式」下启动。

## 三、启动中间件（可选）

### 最小模式（无需任何中间件）

不启动 PostgreSQL / Redis / Milvus，应用仍可启动：
- GET /api/v1/health 返回 {"status":"ok"} — 正常
- GET /api/v1/health/ready 返回 database: down — 表示数据库未连接
- /api/v1/chat 会返回 503（未配置 API Key）或调用 OpenAI
- /api/v1/documents 需要数据库，未连接时会报错

### 全栈模式（Docker Compose）

```powershell
# 确保 .env 中 OPENAI_API_KEY 已填写
docker compose up -d --build
```

Compose 会拉起：app、postgres、redis、milvus（含 etcd、minio）。
首次启动 Milvus 需数十秒就绪。

如果只想启动中间件（不用容器化 app）：

```powershell
docker compose up -d postgres redis etcd minio milvus
```

然后本机直接运行 uvicorn（见第四节），.env 中数据库地址保持 localhost 即可。

## 四、启动应用

```powershell
cd D:\code\ai-agent\project-python
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问：
- 健康检查：http://127.0.0.1:8000/api/v1/health
- 就绪检查：http://127.0.0.1:8000/api/v1/health/ready
- API 文档：http://127.0.0.1:8000/docs

## 五、验证

```powershell
# 健康检查
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health

# 对话（需填写 OPENAI_API_KEY 且能访问 OpenAI）
$body = @{ messages = @(@{ role="user"; content="你好" }) } | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/chat -Method Post -Body $body -ContentType "application/json"
```

## 数据库表初始化（如需文档功能）

本项目未内置自动建表。如需 /documents 接口，需手动建表：

```powershell
.\.venv\Scripts\python.exe -c "import asyncio; from app.infrastructure.database.session import init_engine, configure_session; from app.infrastructure.database.models import Base; from app.config import get_settings; async def main(): engine = init_engine(get_settings().database_url); async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all); print('done'); asyncio.run(main())"
```

执行前提：PostgreSQL 已启动且 agent_db 数据库存在。
若用 Docker Compose 的 postgres，数据库会自动创建。

## 目录速查

| 文件 | 用途 |
|------|------|
| .env | 环境变量配置（由 .env.example 复制） |
| requirements-core.txt | 核心依赖（不含 unstructured） |
| requirements.txt | 完整依赖（含 unstructured） |
| docker-compose.yml | 全栈一键部署 |
| app/main.py | FastAPI 入口 |
| app/config.py | 配置定义 |
| app/infrastructure/ | 数据库、Redis、Milvus、LLM、追踪 |
| app/core/ | Agent 编排、RAG、记忆、意图、工具 |
| app/etl/ | 文档解析与分块 |
| app/api/routes/ | 对话、文档、健康检查路由 |
## 完整运行命令（无注释）

```powershell
cd D:\code\ai-agent\project-python

.\.venv\Scripts\Activate.ps1 // 每次运行前激活
python.exe -m venv .venv // 一次性
python.exe -m pip install -r requirements-core.txt // 一次性
python.exe -m pip install -e . // 一次性
Copy-Item .env.example .env // 一次性
python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 // 每次运行

docker compose up -d --build // 构建全套容器+容器化 app
docker compose start app     // 启动 app 容器
```
