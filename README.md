# evidence-rank-memory · v0.4.0

提供 Add / Search 接口的记忆检索服务。Add 保存对话并抽取事实；Search 在用户范围内融合 BM25、向量、实体、数字、时间及多跳线索，返回记忆证据，由调用方生成回答。可选使用 BGE cross-encoder 对候选结果重排。

最初的 v0.4.0 发布取自服务器 `agent-memory-v0.4.0` 的源码快照；`app/`、`scripts/`、`train/` 和 `eval/` 中的 Python 文件在导入时保持原样。发布不包含运行数据库、密钥、虚拟环境、外部数据集、生成的训练集和模型权重。原始导入快照的文件校验见 `SOURCE_MANIFEST.json`；后续的网关兼容修复见 [恢复说明](docs/RECOVERY.md)。

## 本地启动

建议使用 Python 3.12 或更新版本。

```bash
git clone https://github.com/wonder324/evidence-rank-memory.git
cd evidence-rank-memory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/contract_check.py
python -m app
```

`.env.example` 默认使用本地哈希向量、规则抽取并关闭重排，不需要外部 API 密钥。数据保存在 `data/memory.db`。API 密钥为空时不鉴权；公网部署请在 `.env` 设置自己的 `MEMORY_API_KEY`。

## 接口

- `GET /health`：返回 `{"status":"ok","system":"evidence-rank-memory","version":"0.4.0"}`。
- `POST /add`：保存一个用户会话的消息。
- `POST /search`：按 `user_id` 检索记忆证据。

设置 `MEMORY_API_KEY` 后，写入和检索需发送 `Authorization: Bearer <你的密钥>`，也支持 `X-Api-Key`。健康检查无需鉴权。

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/add \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","user_id":"demo","session_id":"s1","messages":[{"role":"user","content":"我现在住在南京。"}]}'
curl http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","query":"我现在住在哪里？","top_k":5}'
```

Add 的成功响应包含 `success`、`request_id`、`user_id` 和 `session_id`；Search 返回 `data` 数组。检索按用户隔离，`request_id` 用于幂等写入。默认最多并发处理 8 个 Add/Search，请求超限返回 HTTP 429，可用 `MAX_INFLIGHT` 调整。

## 启用外部模型

在 `.env` 中配置：

```dotenv
EMBEDDER=dashscope
DASHSCOPE_API_KEY=替换为你自己的密钥
EMBEDDING_MODEL=text-embedding-v4
LLM_EXTRACT=1
OPENAI_API_KEY=替换为你自己的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

向量 API 失败时，源码默认回退到哈希向量；设置 `EVAL_STRICT_EMBED=1` 可在失败时直接报错。不同向量模型的数据应使用不同的 `DATA_DIR`，避免混用向量。

重排器仅从本地加载模型。先自行下载 `BAAI/bge-reranker-base`：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="BAAI/bge-reranker-base", local_dir="models/bge-reranker-base")'
```

然后在 `.env` 设置：

```dotenv
RERANKER=1
RERANKER_MODEL=models/bge-reranker-base
RERANKER_TOP_N=20
RERANKER_BLEND=0.45
```

模型下载和外部 API 由使用者自行配置。第三方模型与数据集遵循各自的许可证和服务条款，本仓库不重新分发这些资产。

## 测试与训练脚本

`python scripts/contract_check.py` 使用独立临时数据库，覆盖鉴权、Add/Search、幂等、用户隔离、中英文事实和图片内容等契约行为，并关闭外部模型调用。

`python -m train.fit_weights` 在脚本内置的合成对话上拟合检索融合权重，会覆盖 `models/fusion_weights.json`。日常启动可直接使用仓库内已有权重。

`eval/` 下的评测脚本、`train/build_v3_pairs.py` 和 `train/train_v3.py` 保留用于复现实验；涉及 PerLTQA、LongMemEval 等外部数据的脚本需要自行按脚本路径准备数据。生成的 `train/v3/`、`eval/external/` 和运行产物未纳入发布。

## Docker

```bash
docker build -t evidence-rank-memory:0.4.0 .
docker run --env-file .env -p 8080:8080 \
  -v "$(pwd)/data:/app/data" evidence-rank-memory:0.4.0
```

使用重排器时另外挂载模型目录，并将 `RERANKER_MODEL` 设为容器内路径。公网 HTTPS 可参考 `deploy/Caddyfile`，将域名替换为自己的域名。

## 许可证

本项目源码采用 [MIT License](LICENSE)。Python 依赖和可选第三方模型、数据集仍适用各自的许可证。

## 网关兼容与恢复

参见 [恢复说明](docs/RECOVERY.md)。开启 `LLM_EXTRACT_STRICT=1` 后，已启用的模型抽取若失败，将在该次 Add 写入前返回 503，调用方可使用相同 `request_id` 重试。已有成功 Add 的重试会先检查持久化幂等记录，避免重复模型调用。
