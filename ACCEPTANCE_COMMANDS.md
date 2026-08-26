# 验收命令

> 每完成一个功能点，在此追加对应验收命令。改动后重跑，出现预期输出即验收通过。

## M0 · 基础设施

### M0-F1 · Embedding 封装

```powershell
# 在项目根目录执行，首次运行会下载 bge-small-zh-v1.5 模型
.venv\Scripts\python.exe -m pytest tests/infrastructure/embedding/test_embedder.py -v
```

期望输出：`4 passed`。
