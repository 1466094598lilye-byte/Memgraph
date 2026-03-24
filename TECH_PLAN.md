# MemGraph 非 LLM 技术方案

## 技术方案

### 1. Encode：纯本地、零 API

**节点来源（三路并行）**：

| 来源 | 方法 | 说明 |
|------|------|------|
| 实体 | NER（spaCy 多语言） | PERSON, ORG, DATE, MONEY 等 → ENTITY 节点 |
| 关键信息 | Keyphrase（KeyBERT / YAKE） | 每轮抽取 top-k 短语 → 作为信息单元节点 |
| 兜底 | 句子 chunk | 过长或无法抽取时，按句切分并 embed 存为 raw trace |

**边**：
- 同轮共现 → 连边
- 相邻轮（±1）→ 连边，权重略低
- 不依赖 LLM 做关系分类，只做结构连接

**话题**：
- 沿用现有 embedding 聚类，不调用 LLM

**类型简化**：
- 不再区分 intent / constraint / state / completed
- 统一为「信息单元」：有 embedding、可检索即可
- 有 NER 标签的保留为 ENTITY，其余为通用节点

### 2. Activate：检索 + 长度控制

- 保持现有逻辑：query embedding → 节点检索 → 邻居扩展 → 排序
- 增加 raw_traces 检索
- 输出长度控制：max_chars 或 max_tokens，使 |输出| ≤ 0.3 × |完整对话|，与 h2o 对齐

### 3. 与 h2o 的 token 对齐

- h2o：每 query 注入约 30% 对话
- MemGraph：每 query 注入 activate 结果
- 设定：`activate(..., max_output_chars=0.3 * len(full_conv_text))`，保证单 query 的 token 不超过 h2o

### 4. 召回 90% 的保障

- 编码覆盖：NER + Keyphrase 尽量覆盖重要实体和短语；覆盖不到的用 raw traces 兜底
- 检索质量：embedding 检索 + 邻居扩展，保证相关节点被拉到
- 长度与精度的平衡：在 max_output_chars 内优先返回与 query 最相关的节点，避免无关内容占长度

### 5. 依赖（均为本地）

| 组件 | 依赖 | 说明 |
|------|------|------|
| Embedding | sentence-transformers | 已有 |
| NER | spaCy (xx_ent_wiki_sm) 或 jieba | 需新增 |
| Keyphrase | KeyBERT 或 YAKE | 需新增 |

无 LLM API，无 GPU 硬性要求。

### 6. 实施顺序

1. 抽象 Extractor 接口：定义 `extract(chunk) -> (nodes, edges)`，便于切换实现
2. 实现 NonLLMExtractor：NER + Keyphrase + 共现边
3. Activate 增加 max_output 控制：与 h2o 的 token 对齐
4. Benchmark：在相同测试集上对比 token 和 recall，迭代到满足约束

---

## 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| **NER 覆盖不足** | 中 | 实体漏抽，影响 recall | ① 多语言模型（xx_ent_wiki_sm）覆盖 100+ 语言；② 覆盖不到的用 Keyphrase 和 raw traces 兜底；③ 若某语言表现差，可考虑加语言专用模型（如 jieba 中文） |
| **Keyphrase 噪声大** | 中 | 无关短语占节点，稀释检索精度 | ① 调低 top-k，只取高置信度短语；② 可对比 KeyBERT vs YAKE，选噪声更小的；③ 邻居扩展时按相似度排序，噪声节点自然靠后 |
| **召回达不到 90%** | 高 | 核心目标未达成 | ① raw traces 兜底保证「至少能搜到原文」；② 在 benchmark 上迭代：调 top_k、邻居扩展深度、max_output 内排序策略；③ 若仍不足，可考虑在 activate 阶段加轻量 rerank（不违反零 API 约束） |
| **Token 超 h2o** | 中 | 违反北极星约束 | ① max_output_chars 硬截断，优先保证不超；② 在 benchmark 上实测 h2o 的 30% 对应多少字符，对齐阈值；③ 输出按相关性排序，截断时优先保留高相关 |
| **Embedding 语义偏差** | 低 | 检索不到相关节点 | ① 沿用 paraphrase-multilingual-MiniLM，已覆盖多语言；② 若特定领域偏差大，可换 domain 相近的模型（仍本地）；③ 邻居扩展弥补单点检索遗漏 |
| **共现边噪声** | 低 | 无关节点被拉入 | ① 边只做结构连接，不做关系语义；② 检索时按 query 相似度排序，噪声节点靠后；③ 在 max_output 内自然被截断 |
| **spaCy / KeyBERT 依赖重** | 低 | 安装慢、包体积大 | ① 文档化 requirements，可选 lightweight 方案（如 YAKE 比 KeyBERT 轻）；② 首次加载懒加载，不阻塞启动 |
| **话题聚类质量下降** | 中 | 话题路由不准，检索分散 | ① 沿用现有 embedding 聚类，不依赖 LLM；② 若聚类过碎，可调相似度阈值合并；③ 话题数 < 2 时退化为全局检索，保证下限 |
| **Benchmark 对齐歧义** | 中 | h2o 的 30% 定义不清 | ① 在相同测试集上实测 h2o 的 token 消耗；② 以实测值为准设定 max_output；③ 文档化对齐方法，便于复现 |

### 风险优先级

1. **P0**：召回 90%、Token ≤ h2o —— 直接对应北极星，必须通过 benchmark 迭代达成
2. **P1**：NER/Keyphrase 覆盖、max_output 截断策略 —— 影响 P0，需优先调优
3. **P2**：依赖体积、话题聚类 —— 可接受一定折中，后续优化
