# 来源与方法变更披露

2026-10-29 补充。以下列出能够从当前仓库与记录确认的技术组件，不是完整原创性证明。若存在未记录的上游代码复用，仍需作者补充原作者、原始报告、链接、许可证和具体修改文件。

| 组件/资产 | 来源及本项目使用方式 |
| --- | --- |
| BGE 重排 | BAAI/bge-reranker-base：https://huggingface.co/BAAI/bge-reranker-base 。本地 CrossEncoder 候选窗口评分，再与检索次序融合。历史训练脚本不证明正式部署使用微调模型；实际权重 revision/哈希待核实。 |
| 向量服务 | DashScope text-embedding-v4。运行配置与异常 hash 回退边界见 EVALUATION_V045.md。 |
| LLM 抽取 | 请求模型 gpt-4o-mini，实际经过 API2D；提示词、解析和有限重试公开于 app/llm_extract.py。供应商后端路由未经独立验证。 |
| 检索方法 | 使用 BM25、向量融合、规则特征、MMR、二跳候选和交叉编码器；描述的是当前实现，不主张这些基础方法由本项目首创。 |
| 第三方软件 | FastAPI、Uvicorn、Pydantic、HTTPX、sentence-transformers 等，见 requirements.txt。各项目许可证独立适用，项目 MIT 不替代它们。 |
| PerLTQA 开发实验 | eval/perltqa_*、train/build_v3_pairs.py 等读取本地公开数据文件；原始数据未分发。精确来源版本、许可证和训练/验证划分仍需作者补全。 |
| LongMemEval 开发实验 | https://github.com/xiaowu0162/LongMemEval 。eval/longmemeval_compare.py 等作为离线实验工具保留；数据与模型资产未分发。精确版本及与比赛数据的关系需核实。 |
| 融合权重 | models/fusion_weights.json 与 train/fit_weights.py 中的合成训练示例；这些小规模指标不是正式比赛结果。 |

## 公开实现中的方法变化

- 原始公开快照：v0.4.0，提交 fffd6f0b29de873142706e0007937c7d9eaeef2a。
- v0.4.1：网关 JSON 包装兼容、可选严格抽取、幂等检查与脱敏日志，提交 fedc57c1ccea0585fa9b00cef6c289586bb7debc。
- v0.4.2–v0.4.3：继续增强抽取响应解析与有限恢复；期间存在抽取 API 网关切换，不能描述为完全无行为变化。
- v0.4.4：Add 捕获抽取失败后保存并索引源消息，属于失败处理行为变化。
- v0.4.5：缩小数据库锁范围，新增有界 Search 流水线并发和状态耗时日志。离线对比结果相同不证明正式任务中无影响。

这些中间版本未各自形成当前仓库中的独立固定发布提交，完整历史需结合部署备份。是否获得正式 Full 期间的变更批准尚未确认。

## 数据与独立性边界

评测请求、答案、数据库、备份和返回正文不属于开源交付。历史代码包含公开基准开发实验工具，不能仅凭这些文件证明数据无重叠，也不能据此认定存在泄漏。正式服务调用路径未发现读取这些离线 gold 数据文件，但完整数据来源与训练资产仍需作者核验。UNREAL/Qwen 探针实验未见于当前运行服务调用路径，不能写成已部署方法。
