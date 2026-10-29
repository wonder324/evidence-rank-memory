# v0.4.5 方法与运行说明

本说明于 2026-10-29 根据公开源码、部署记录和只读配置核对补充，不构成平台批准证明，也不追溯修改已提交的版本。

## 固定源码与方法

运行源码提交：`579cafca5fad7d4993595de6f7c7420c958b5b88`。

Add 按输入顺序保存消息，提取规则事实和可选 LLM 事实，维护用户内的事实状态与索引。LLM 仅在 Add 抽取阶段使用；当前提示词和响应修复逻辑位于 `app/llm_extract.py`。抽取失败后持久化和索引原始消息，成功响应在事务提交后发出。失败提取不保证获得结构化事实。

Search 在同一 user_id 内融合 BM25、向量、实体、数字、时间及规则特征，进行二跳召回、去冗余与可选交叉编码器重排。options 用于候选相关性评分，不是金标。返回记录的 id/content/score/created_at，不调用生成式 LLM 输出最终答案。模型重排仍串行保护；SEARCH_MAX_ACTIVE 限制进入检索流水线的任务数。MAX_INFLIGHT 是 Add/Search 入口接纳上限，两者不是实际并行模型推理数。

## 2026-10-29 核对的非敏感配置

下列内容是核对时的运行配置，不是整个历史任务的逐时证明；路径采用可迁移写法，凭据不公开。

```dotenv
EMBEDDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
LLM_EXTRACT=1
LLM_EXTRACT_STRICT=1
OPENAI_BASE_URL=https://oa.api2d.net/v1
LLM_MODEL=gpt-4o-mini
RERANKER=1
RERANKER_MODEL=models/bge-reranker-base
RERANKER_TOP_N=20
RERANKER_BLEND=0.45
RERANKER_CONTEXT=0
CHRONOS=0
MAX_INFLIGHT=32
OMP_NUM_THREADS=1
```

核对时未在配置白名单中显式设置 SEARCH_MAX_ACTIVE，v0.4.5 源码默认值为 4。未显式设置 EVAL_STRICT_EMBED；当前代码允许向量请求失败后回退 hash。是否在正式 Full 触发过回退尚未确认，不能将该事实写成已满足指定向量模型要求。未来启用严格失败是运行配置变更，不能追溯改变历史评测。

抽取实际经过 API2D 代理，请求模型名 gpt-4o-mini；本仓库不能独立证明供应商实际路由模型或其历史版本。此项需供应商记录和主办方确认。

正式部署使用 2 vCPU、约 2 GB RAM 和 2 GB swap，重排模型本地加载。融合权重文件为仓库 models/fusion_weights.json，SHA-256 为 d38ab7db49e80040368a4a5ca9f2c45f604e7e1fae43dd95519e92cde6a271f9；其中 embedder 元数据为 hash，表示该文件的训练配置，不会覆盖运行时 EMBEDDER。

## 隔离复现

```bash
git clone https://github.com/wonder324/evidence-rank-memory.git
cd evidence-rank-memory
git checkout 579cafca5fad7d4993595de6f7c7420c958b5b88
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m unittest discover -s tests
python scripts/contract_check.py
```

上述为离线功能检查。模型运行需按 README 下载 BGE 权重，使用自己的 API 凭据及独立 DATA_DIR，再手工按上表配置；禁止使用或分发原评测数据库。不要在复现时运行 train.fit_weights 覆盖固定融合权重。合成契约测试通过不等于重现正式分数。

依赖文件使用版本范围，尚无历史环境锁文件；BGE 权重也未记录已验证的固定 revision 和哈希。本说明不虚构版本锁或模型校验值，因此目前只能说明方法与配置，不能保证位级或分数级复现。
