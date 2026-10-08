# Agent Memory Challenge 参赛服务

文本、代码、多模态共用这一套 Add / Search。平台负责回答和评分，这里只返回记忆证据。

## 本地跑起来

```bash
cd agent-memory-challenge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m train.fit_weights
python scripts/contract_check.py
python -m app
```

健康检查是 `GET /health`。写入是 `POST /add`，检索是 `POST /search`。报名时填写公网地址，例如 `https://你的域名/add` 和 `https://你的域名/search`。

组别是开源方法榜。向量用 `text-embedding-v4`。Add 阶段的事实抽取必须调用真实的 `gpt-4o-mini`：在 `.env` 填入 `OPENAI_API_KEY`，`LLM_MODEL` 保持 `gpt-4o-mini`。密钥留空时只有规则抽取，不能拿去报名。不要把百炼密钥或 `qwen-plus` 写成这个模型。

换了向量模型之后重新执行 `python -m train.fit_weights`。评测流量不能用来训练，任务结束后 30 天内执行下面的命令删数据：

```bash
python -c "from app.config import get_settings; from app.embedder import Embedder; from app.store import MemoryStore; s=get_settings(); MemoryStore(s, Embedder(s), s.weights()).purge_all()"
```

## 报名时可以直接粘贴的说明

系统名称：`evidence-rank-memory`  
版本：`0.1.0`  
赛道：文本记忆  
组别：开源方法榜  
鉴权：Bearer，密钥等于部署环境里的 `MEMORY_API_KEY`（见本机 `.env`，不要把 `.env` 提交进仓库）  
接口：`https://zyd324.online/health`、`https://zyd324.online/add`、`https://zyd324.online/search`

方法说明：参赛系统只提供 Add 与 Search。Add 同步保存原始对话，并用规则加 gpt-4o-mini 抽取居住地、姓名、职业、偏好、规则和事件；单值事实被新值替换后仍保留旧值。向量模型是 text-embedding-v4。Search 在 `user_id` 内做 BM25、向量、实体、数字和时间意图融合，必要时按实体做二跳，返回记忆证据，不生成答案。权重由合成探针拟合，没有使用评测流量。代码在公开仓库的固定 commit。未封装 Mem0、Zep 或其他未注明的仓库。

公开仓库：https://github.com/wonder324/evidence-rank-memory

容量：单进程，默认同时处理 8 个 Add/Search；超出返回 HTTP 429，`Retry-After: 2`。正式评测 `top_k` 为 100。单个用户记忆不超过 2500 条时全量打分，超过后用 FTS 取最多 800 条候选，并保留当前有效事实。图片单张不超过 10MiB，一次 Search 返回的图片总量不超过 24MiB。平台单次请求上限 30 分钟。

原创声明：本系统为本次赛事独立实现，使用公开的 BM25 与向量检索思路。未封装 Mem0、Zep 或其他未注明的仓库。若后续复用论文或代码，必须在申请里补上原作者、技术报告和改动。

部署示例：

```bash
docker build -t evidence-rank-memory .
docker run --env-file .env -p 8080:8080 evidence-rank-memory
```

公网机器上还需要一层 HTTPS。平台拒绝 localhost、内网地址，以及写在 URL 里的账号密码。

检查还缺什么：

```bash
python scripts/check_ready.py
```
