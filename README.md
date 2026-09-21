# Agentic RAG Hub — 企业级知识库实践说明

目标：搭一条**可上线的中文企业知识库**流水线——带权限、可增量、可追溯、可回归评测；检索与存储按配置插件化，避免业务代码绑死某一家向量库。

嵌入：`BAAI/bge-m3`（1024 维）。重排：`BAAI/bge-reranker-v2-m3`。生成：DeepSeek（OpenAI 兼容）。GraphRAG / 多 Agent 非主路径（见 §8.8）。

---

## 目录

1. [项目定位](#1-项目定位)
2. [心智模型：五层与赛道](#2-心智模型五层与赛道)
3. [原理总览](#3-原理总览)
4. [端到端步骤与阶段路线](#4-端到端步骤与阶段路线)
5. [术语表](#5-术语表)
6. [技术栈与实现状态](#6-技术栈与实现状态)
7. [插件选型](#7-插件选型)
8. [分阶段实践（原理 · 流程 · 验收）](#8-分阶段实践原理--流程--验收)
9. [仓库结构](#9-仓库结构)
10. [安装](#10-安装)
11. [快速上手](#11-快速上手)
12. [配置要点](#12-配置要点)
13. [评估与指标](#13-评估与指标)
14. [换项目检查清单](#14-换项目检查清单)
15. [常见问题与坑](#15-常见问题与坑)
16. [当前局限与下一步](#16-当前局限与下一步)

---

## 1. 项目定位

### 1.1 要建成什么样的系统

| 能力 | 含义 |
| --- | --- |
| 混合检索 | BM25（专名/数字）+ 稠密向量（同义改写）→ RRF 融合 → 重排精排 |
| 权限（ACL） | 检索阶段过滤；无权限 ≈ 库中不存在，而不是生成后再删 |
| 增量入库 | 按文档 `content_hash` 判断增/改/删；账本记录每次作业 |
| 约束生成 | 无证据则拒答（「信息不足」）；答案带出处 |
| 可观测 / 可回归 | Langfuse（可选）+ 人写黄金集；公开 CRUD 评测轨另立 |

### 1.2 Demo RAG vs 企业 RAG

| 问题 | Demo | 本仓库取向 |
| --- | --- | --- |
| 谁能看见文档？ | 谁搜都能中 | Chunk 带 `acl[]`，索引内过滤 |
| 制度改了？ | 全量重嵌 | 账本 + hash，只处理变更；删文件同步删索引 |
| 答案从哪来？ | 「很像」的一段 | 出处可点回 source / section / page；无证据拒答 |
| 表格被切开？ | 固定 512 字切 | 先结构切片，表整块保留（解析器到位后） |
| 怎么知道变好了？ | 肉眼看两条 | **人写黄金集**回归；RAGAS 仅辅助 |

「能跑」≈ §8.0～8.3；「像企业」≈ ACL、拒答、黄金集、Demo 三动作（§8.2 / 8.4 / 8.6 / 8.7）。

### 1.3 两条轨勿混

| 轨 | 配置 | 语料 / 评测 | 生成 |
| --- | --- | --- | --- |
| 企业 ACL 沙盒 | `config/rag.yaml` | `data/sandbox/*.md` + `data/eval/golden.jsonl` | 拒答 + 出处 SYSTEM |
| CRUD 公开评测 | `config/rag.crud.yaml` | `data/corpus/80000_docs` + `crud_split/` + `scripts/eval_crud.py` | 官方任务 prompt（`src/eval/crud/prompts/`） |

两套 **index / collection / `ledger.tenant_id`** 必须隔离。勿把新闻 CRUD 当成内网制度讲权限。

---

## 2. 心智模型：五层与赛道

把文档扔进向量库 ≠ RAG。能上线的系统至少打通五层：

| 层 | 解决什么 | 翻车例子 |
| --- | --- | --- |
| **1 文档解析** | 扫描件、跨页表、骑缝章、嵌图文字 | 表头与表体分家 → 答出「既往症等待期 9 天」 |
| **2 切片策略** | 按标题还是语义、长度、重叠 | 切错后检索内容与问题永远对不齐 |
| **3 检索策略** | 向量 / 关键词 / 混合、重排、是否改写 | 只靠一路向量 = 上线开盲盒 |
| **4 生成控制** | 怎么喂模型、压幻觉、是否「只准据资料答」 | 用户用一次就弃用 |
| **5 评测反馈** | 召回准不准、Bad Case 怎么回流 | 无评测 = 闭眼开车 |

铁律：**解析层不过关，后面一律不许开工。**

会拖 Dify ≠ 能交付：框架是入门券；饭碗是检索原理、Embedding 选型、解析天花板、业务 Prompt、长在坏例子上的评测。标准答案若全由 GPT 自编自判，98 分不可信。

赛道分层（现实感）：拖拽 Demo 易被卷；交付级要懂解析/检索/Prompt/评测；架构级啃 GraphRAG / Agentic / 多模态硬骨头。下半场分水岭不在「能不能跑通」，而在复杂文档与复杂问题接不接得住。

建议：从**检索原理**入门；至少做一个有真实用户与 Bad Case 回流的项目；有余力再往 GraphRAG / Agentic 靠（本仓库主路径仍是混合检索 + 约束生成）。

---

## 3. 原理总览

### 3.1 一句话

**解析 → 切片 → 赋元数据 → 双写索引 → 鉴权召回 → 融合精排 → 约束生成 → 评测回归。**

稀疏索引和向量库都是**插件**：业务只依赖统一接口，引擎用配置切换。

### 3.2 检索四段论

1. **表示**：词袋（BM25）或向量（BGE-M3，1024 维）
2. **召回**：两路各取 Top-N，求「别漏」
3. **排序**：RRF 合路 + reranker 精排，求「别乱」
4. **生成**：只根据证据写答案，求「别编」

向量擅同义，BM25 擅专名/数字。两路原始**分数量纲不同，不能直接相加**；RRF 只看排名。Rerank 贵，只对召回后约 10～20 条跑。

### 3.3 为何嵌入 / 重排走硅基流动

本机加载 BGE 权重（尤其 Intel Mac）成本高、HF 网络不稳。硅基提供同名 HTTP API：算法故事不变，工程上变成「有 Key 就能嵌」。企业常见形态是「推理网关托管检索模型」。限流见 `src/llm/rate_limit.py`（`RateBudget`：调用前触顶等待，调用后按响应 token 记账；遇 429 指数退避）。

| 能力 | 模型 / 接口 | 说明 |
| --- | --- | --- |
| 嵌入 | `BAAI/bge-m3` → `/v1/embeddings` | 1024 维；单条约 8192 token 量级 |
| 重排 | `bge-reranker-v2-m3` → `/v1/rerank` | **不是** Chat；入参 `query` + `documents` |
| 生成 | DeepSeek 等 `/v1/chat/completions` | 默认 DeepSeek |

查询侧加前缀（文档侧不加）：`为这个句子生成表示以用于检索相关文章：`。漏加或两侧都加 → 坐标系错位，召回掉一截。

### 3.4 插件边界与协议

| 协议 | 职责 |
| --- | --- |
| `IngestSource` | 发现逻辑文档（`local_glob` 每文件一篇；`line_corpus` 每行一篇） |
| `Parser` + `Chunker` | `ingest.routing` 按扩展名选择 |
| `SparseIndex` / `VectorIndex` | BM25 侧 / 向量侧；`search` 内做 ACL |
| `IngestLedger` | 增量账本（MySQL 或本地 JSON） |

统一形状（核心字段）：

- `Chunk`：`chunk_id, text, source, page, section?, acl[], content_hash`
- `Hit`：`chunk_id, text, source, score, acl, page`
- RRF 对齐键 = **`chunk_id`**（禁止用 Python `id()` 或仅全文）
- ACL：`acl ∩ roles` 非空，或 `acl` 含 `"all"`；过滤在插件内，生成前不得泄漏进 prompt
- `VectorIndex` 只收已 embed 的向量，**内部不调硅基**

未知 `backend`：**启动即 raise**，禁止静默回退 jsonl。换 ES / Milvus / Qdrant = 改 yaml，不改 `hybrid_search`。

---

## 4. 端到端步骤与阶段路线

```
① 解析        ingest.source 列逻辑文档 →（文件走 Router / 行语料直接切片）
② 切片        Chunker（如 markdown_heading / fixed_size）
③ 元数据      source / page / acl / content_hash / chunk_id
④ 双写        SparseIndex.upsert + VectorIndex.upsert + Ledger
⑤ 检索        稀疏 ∥ 向量 → 分数过滤 → RRF → rerank → top_k（ACL 在插件内）
⑥ 生成        仅据资料；否则拒答；列出处（CRUD 轨换任务 prompt）
⑦ 编排        LangGraph：rewrite → retrieve → generate
⑧ 评测上线    黄金集 / CRUD 指标 +（可选）Langfuse + Gradio Demo
```

| 阶段 | 主题 | 建议日程 | 产出 |
| --- | --- | --- | --- |
| 0 | Key、探活 | D1 上午 | `probe.py` 通硅基+稀疏/向量/账本 |
| 1 | 解析切片 | D1 下午 | 沙盒 md → Chunk |
| 2 | 入库 + ACL + 账本 | D2 | 双写 + 增量跳过 |
| 3 | 混合检索 | D3 | `hybrid_search` |
| 4 | 拒答与出处 | D4 | E2E 问答 |
| 5 | LangGraph | D5 | `invoke_rag` |
| 6 | 黄金集 | D6 | `golden_report` |
| 7 | Demo | D7 | Gradio 三动作 |
| 8 | ES / 图谱 / RAGAS | 之后 | 可选增强 |

---

## 5. 术语表

| 术语 | 含义 |
| --- | --- |
| Chunk | 检索与入库的最小文本单元，含正文与元数据 |
| ACL | 访问控制标签；与用户 `roles` 有交集或含 `all` 才可见 |
| BM25 | 经典词项检索；ES / 内存 jieba+rank_bm25 |
| 稠密向量 | 语义嵌入；本仓库固定 1024 维（BGE-M3） |
| RRF | Reciprocal Rank Fusion，按排名融合多路召回 |
| Rerank | 查询-文档成对精排（bge-reranker） |
| content_hash | 文档内容指纹；未变则跳过重嵌 |
| Ledger | 入库账本：`kb_document` / `kb_ingest_job` |
| Source | 语料发现层（≠ Parser） |
| Router | 按扩展名把文件路由到 Parser/Chunker |
| 黄金集 | 人写问答与期望（拒答/必含词/角色），用于回归 |
| GT | 标准答案。模型写出来的文字，就是拿它来对比打分 |
| BLEU | 看「用词像不像」：生成里有多少词（及连续词组）也出现在标准答案里。越高越像抄对了原词 |
| BP | 过短扣分。答太短时标准 BLEU 会罚分；本仓库默认把这个罚分除掉再报（对齐 CRUD 官方） |
| ROUGE-L | 看「句子骨架像不像」：两边按顺序能对上的最长一段有多像。摘要、续写常看它 |
| bertScore | 看「意思像不像」：换了说法也能给分（本仓库用 text2vec，不是 HuggingFace 那套 bert-score） |
| QA_avg_F1 / QA_recall | 看「事实盖住没有」：先从标准答案里出几道小题，再用生成文去答。能答上几道是 recall，答得像不像是 F1 |
| length | 生成了多少个字。用来发现空答、一句话敷衍、或啰嗦过长 |
| RAGAS | Retrieval Augmented Generation Assessment。让另一个大模型当裁判，看回答是否忠于检索资料、检索段落是否相关（如 Faithfulness / Context Precision）。**不能替代**人写黄金集；本仓库脚本仍占位 |
| 消融 | 故意关掉系统某一块（如 `--baseline no_retrieve` 关掉检索），再跑同一套题，看分数掉多少，用来证明「这一块有没有用」 |
| CRUD-RAG | 公开中文基准（续写/摘要/纠错/问答）；评测轨另配 |

同一条对照（标准答案：「上海市发放了 500 万元体育消费券」）：

| 指标 | 生成 A：「上海发放了 500 万体育消费券」 | 生成 B：「今天天气不错」 |
| --- | --- | --- |
| BLEU | 高：500 万、消费券等词对上了 | 接近 0：几乎没有相同的词 |
| ROUGE-L | 高：顺序也大致一致 | 低 |
| bertScore | 高；若写成「沪上发了体育优惠券」，字面 BLEU 会掉，这项仍可能不低 | 低：意思完全不同 |
| QA_recall | 题如「发了多少钱？」「什么券？」多半能答 | 多半答「无法推断」，recall 低 |
| length | 只用来看长短，不表示对错 | 同样只看长短 |

BP 例子：生成只写「上海」两个字，词全对但太短。带罚分的 BLEU 会被压低；本仓库默认报的是去掉罚分后的分，所以**不要和网上带 BP 的 BLEU 表直接比**。

---

## 6. 技术栈与实现状态

以代码为准（`NotImplementedError` = 尚未实现）。

### 6.1 已实现（可跑通主路径）

| 层 | 选项 |
| --- | --- |
| 嵌入 / 重排 | 硅基 BGE-M3 / bge-reranker + `RateBudget` |
| 生成 | DeepSeek（OpenAI SDK） |
| 语料源 | `local_glob`（沙盒）/ `line_corpus`（CRUD 一行一篇） |
| 解析 | `markdown`（frontmatter ACL） |
| 切片 | `markdown_heading`、`fixed_size`（CRUD 128/0） |
| 稀疏索引 | `memory_jsonl`、`elasticsearch`（无 IK 时回退 standard）、`postgres_fts`（jieba+tsvector） |
| 向量索引 | `qdrant`、`milvus`、`pgvector` |
| 账本 | `mysql`（SQLAlchemy + Alembic）、`json` |
| 编排 | LangGraph 三节点 |
| 评测 | 沙盒 `eval_golden.py`；CRUD `eval_crud.py`（BLEU/ROUGE，可选 bert/QuestEval） |
| UI | Gradio（`src/app.py`） |
| 可观测 | Langfuse Callback（密钥走环境变量） |

### 6.2 配置已预留、代码未完成（占位）

| 项 | 状态 |
| --- | --- |
| Parser：`docling` / `llamaparse` / `unstructured` | 占位 |
| Chunker：`pdf_layout` / `table_row` / `faq_pair` | 占位 |
| Sparse：`qdrant_sparse` | 占位 |
| RAGAS / 消融脚本 | 占位（先稳定黄金集） |
| SSO / 审计 / OCR 主路径 / 多租户账单 / Bad Case 在线回流 | 未做 |

### 6.3 层选型一览（可换什么）

| 层 | 默认 | 可换 | 动哪里 |
| --- | --- | --- | --- |
| 嵌入 | 硅基 bge-m3 | 同维其他 | `.env`；维变重建集合 |
| 重排 | 硅基 reranker | 其他 `/v1/rerank` | `.env` |
| 生成 | DeepSeek | OpenAI 兼容 | `.env` |
| 解析 | Markdown + 扩展名路由 | Docling / LlamaParse / Unstructured（占位） | `ingest.routing` |
| 切片 | markdown_heading | pdf_layout / table_row / faq_pair（占位）；CRUD 用 fixed_size | `routing` / `chunking` |
| 语料源 | local_glob | line_corpus / 将来 S3·上传 API | `ingest.source` |
| 向量 | qdrant / milvus / pgvector | — | `vector.backend` |
| 稀疏 | memory_jsonl / ES / postgres_fts | qdrant_sparse（占位） | `sparse.backend` |
| 编排 | LangGraph | 一般不换 | — |
| 可观测 | Langfuse 云 | 自托管 | `.env` |

最小依赖沙盒：**Qdrant + MySQL（或 json 账本）+ memory_jsonl**。打开本地 yaml 核对当前默认组合。

落地顺序建议：① 协议+工厂+`memory_jsonl`+`qdrant`+`markdown` → ② 黄金集锁 ACL/RRF/切片 → ③ 再加 docling / ES / milvus；**第一周勿接四套引擎**。

---

## 7. 插件选型

### 7.1 为何插件，而不是「先 jsonl 以后再改」

`hybrid_search` 若写死 `QdrantClient` / 读死 jsonl，换引擎就要重写检索层。三套组合下 **RRF / ACL / chunk_id / embed·rerank 必须相同**；变的只是「倒排放哪、向量放哪」。`memory_jsonl` = 全量进内存扫库，**万级以上换倒排**，勿把单文件堆到 GB。

### 7.2 稀疏插件怎么选

| `sparse.backend` | 原理 | 正文在哪 | 适合 |
| --- | --- | --- | --- |
| `memory_jsonl` | jieba + 内存 BM25Okapi | 一个 jsonl | 沙盒 / 黄金集 / 默认起步 |
| `elasticsearch` | Lucene BM25，分片增量 | ES `_source` | 万级+中文生产（**须 IK**） |
| `postgres_fts` | jieba 切词 + `tsvector` / OR `tsquery` + `ts_rank` | PG 表 text + tsv | 已有 Postgres、不想上 ES |
| `qdrant_sparse` | Qdrant sparse | 同 collection payload | 少维护一套 ES（**占位**） |

### 7.3 向量插件怎么选

| `vector.backend` | 原理 | 适合 |
| --- | --- | --- |
| `qdrant` | 余弦 + payload filter | 默认推荐；本机轻 |
| `milvus` | 大规模 / 分区；standalone 注意内存 | 企业已有 Milvus；CRUD 全量常见 |
| `pgvector` | PG `vector` 扩展 + HNSW（cosine） | 与 `postgres_fts` 共一台库 |

payload 对齐 Chunk；维变 → 新建 collection/表。

### 7.4 解析器怎么选（PDF 上线前必读）

三者都把「版式」变成带结构的文本。公开榜单互相打架，**不要按总分选产品**，用你最脏的 20 页对拍。

| | Unstructured | Docling | LlamaParse |
| --- | --- | --- | --- |
| 出身 | ETL，开源+云 | IBM 开源 | LlamaIndex 托管 API |
| 思路 | 版式→OCR→元素列表 | 版式+表结构→MD/JSON | 整页多模态→LLM 友好 MD |
| 输出 | **元素级**（类型/坐标/页码） | 结构树+MD，多栏较好 | 干净 MD |
| 格式面 | **最宽** | PDF/Office 为主 | 难 PDF，非万能 ETL |
| 部署 | 可自托管；`hi_res` 慢重 | **本地/内网友好** | **云端按页计费**，合规敏感 |

口诀：Unstructured=传送带；Docling=要结构+可私有化（**本仓库上 PDF 时的默认推荐**）；LlamaParse=难页外包。

- Unstructured：`fast` 几乎不做版式；表/多栏常弱；适邮件+PPT+HTML+PDF 混湖
- Docling：表/多栏口碑好；长尾弱；扫描看 OCR；CPU（Intel Mac）慢
- LlamaParse：脏扫描/嵌套表/跨页续表省心；保险合同常过不了出境
- **共同短板**：骑缝章、扫描表头跨页、合并单元格 → 换工具只降概率；抽几页人工看
- 未知 parser/chunker：**对该文件失败**，禁静默纯文本糊弄 PDF
- 当前代码：仅 `markdown` 已实现；docling / llamaparse / unstructured **占位**

### 7.5 切片策略（先结构，再开窗）

没有全局最佳 `chunk_size`。先保证解析带**标题 / 表格 / 页码**。

- **禁**一上来整篇 `RecursiveCharacterTextSplitter(512)`（切开「第 3.2 节」与表）
- 优先按 `#` / Title / section；超长节再 400～800 字（或 256～512 token）重叠窗
- **表整块保留**（企业最值钱）：表头+表体同 chunk；可「小节说明+整张 MD 表」；**勿再按 512 横切表**
- 块首路径：`《员工手册》> 第五章 考勤 > 5.3 加班`
- 用黄金集收口，勿凭感觉 512→384；归因顺序：解析→切块→召回→精排→生成

| 场景 | 倾向 | 原因 |
| --- | --- | --- |
| 制度条款、FAQ | 偏小（一款一条） | 问句对齐单规则 |
| 综述、背景 | 偏大或父子索引 | 要上下文 |
| 表、代码块 | 整块，可略超 | 切开就错 |
| 生成 context | 可比检索块更大 | Parent-Child |
| CRUD 对照官方 | `fixed_size` 128/0 | 与论文设置对齐，勿与沙盒混 |

经验：检索块 300～800 字；overlap ~10%～15%（512/64）。LangChain `chunk_size=512` = **字符非 token**。metadata ≥ `source, page, section, acl, chunk_id`；ACL 文件级继承。

沙盒示例映射：`hr_handbook→[hr,all]`；`finance_reimburse→[finance]`；`public_faq→[all]`。

---

## 8. 分阶段实践（原理 · 流程 · 验收）

运维命令集中在 [§10](#10-安装) / [§11](#11-快速上手)。本节只写**为什么、怎么走、怎么验收**。

### 8.0 环境与硅基

- **原理**：余弦≈夹角；Qdrant 用 `Distance.COSINE`；换库度量须一致。召回宽、精排窄（约 10～20 → 5）。
- **流程**：Key → `.env` → 起向量库 → embed 断言 1024 → rerank 冒烟（「苹果」相关在前）→ `probe.py`。
- **要点**：embed 批 16～64；按 `data[].index` 排序；重排走 `/v1/rerank` 不是 Chat；macOS 连 Qdrant 建议 `trust_env=False` + `127.0.0.1`。
- **验收**：`len(vec)==1024`；rerank 排序合理；`probe` 全绿。

### 8.1 文档解析与切片

- **原理**：解析错则全错。PDF 表跨页拼错是常见事故。Parser 不负责 BM25/向量/RRF。512/64 只是纯文本起点。
- **流程**：文件 → Parser → 结构原料 → Chunker（标题→表整块→开窗+路径）→ `List[Chunk]`。
- **验收**：
  - [ ] Chunk 有 `chunk_id/text/source/acl`；有标题块以「文件>章节」开头
  - [ ] memory_jsonl：行数>0，字段一致
  - [ ] 人工 3 chunk：无半截表、无「见上表」孤儿
  - [ ] 同文件 `content_hash` 同、`chunk_id` 异
  - [ ] 未实现 parser：失败跳过，非纯文本糊弄
  - [ ] （上 PDF 后）抽 3 页对拍多栏/表头表体

### 8.2 带元数据入库（ACL / 增量）

- **原理**：ACL 在 `search` 内过滤。增量靠文件 hash。默认 MySQL 账本（`KbDocument` / `KbIngestJob`）；hash 变则两路 `delete_by_source` + `upsert`。无 MySQL 可 `ledger.backend=json`。
- **流程**：`build_*` → `list_active` → 比 hash → 仅变更 `embed(is_query=False)` 双写 → 删失踪 source → 打印新增/更新/跳过/删除/失败。
- **验收**：
  - [ ] payload / `_source` 有 acl
  - [ ] 稀疏条数与向量条数一致（或可解释的短暂偏差）
  - [ ] `roles=["employee"]` 搜财务 → 空或仅 `all`
  - [ ] 二次 ingest「跳过 N」，embed 调用≈0
  - [ ] 未实现 backend → 启动失败

### 8.3 混合检索：BM25 + 向量 + RRF + 重排

```
宽召回：BM25（词是否出现）∥ 向量（意思像不像）
        → 各路 score 过滤 → RRF(k=60) → reranker（只打候选）→ top_k
```

- BM25：稀有词+频次；中文 jieba / ES IK；分数无统一 0～1 上限
- 向量：文档无前缀；查询加 QUERY_PREFIX；余弦约 0～1
- RRF：`1/(k+rank)`，k=60；**禁** BM25+cosine 直接加
- 过滤：`retrieve.sparse_score` / `vector_score` / `rerank_score` 各管各（gt/gte/lt/lte/relative_gte/keep_at_least）；**不对 RRF 设阈**；数字用黄金集扫
- 经验：`recall_k` 10～20（CRUD 常用 16），`top_k`≈5（CRUD 常用 8）
- **验收**：专名 BM25 中；同义向量中；双路命中更靠前；rerank 后仍无越权；无 Key/超时明确报错（可 `--skip-rerank`）

### 8.4 生成控制：拒答与出处

三层约束（企业轨）：

1. SYSTEM：只据【参考资料】；不足→「信息不足」；禁编造
2. 结构化出处 `[1]` ↔ source(+page)
3. `top_k==0` **短路不调 LLM**

temperature≈0.1。UI 出处以**检索 hits**为准，勿信模型自编书名。CRUD 轨**不要**套这套拒答 SYSTEM，见 §13.3。

- **验收**（人工约 10 条）：有事实能答+真 source；无事实拒答无编数字；无权限 hits 空；明显幻觉 ≤1～2/10。

### 8.5 LangGraph 编排

- **原理**：有限状态机，非自由规划（控延迟/账单）：`rewrite → retrieve(含 ACL) → generate → END`。hits 空 → generate 直接拒答。
- rewrite：补略/纠错，**勿改意图**；无多轮可恒等。
- 可选 route：含「关系/为什么/对比」走图谱（§8.8）；关键词即可。
- 状态：`query, rewritten, roles, hits, answer`。
- **验收**：`invoke_rag` 返 answer+hits；Langfuse 可见三段耗时；换 roles 则 hits 变。

### 8.6 黄金集评测

见 [§13.2](#132-沙盒黄金集企业-acl-轨)；流程与指标用途在评估章展开。

验收：≥20 条含 answer/refuse/forbidden；有通过率与 `badcases.md`；改切片或 ACL 后可对比。

### 8.7 Demo 与上线

必须能点的三个动作：

1. `employee` 问机密制度 → 拒答  
2. `hr` 问同一句 → 答出 + 出处  
3. 改 sandbox 后增量 ingest → 答案或日志证明已更新  

Demo **不**加载本地 BGE。对外说明：沙盒虚构；CRUD 是公开集；生产仍缺 SSO/审计/OCR。

### 8.8 图谱、ES 增强与 RAGAS（可选）

**数据集怎么选（勿整库灌）：**

| 类型 | 是什么 | 适不适合 |
| --- | --- | --- |
| 现成三元组 KG | 头-关系-尾已标 | **抽极小子集**进 Neo4j |
| 多跳文档问答 | 跨 2～3 篇证据 | 对比 hybrid vs graph，非现成图 |

优先级：① 沙盒 `data/eval/org_triples.jsonl`（与 sandbox 对齐）→ ② KgCLUE / CCKS / NLPCC-MH **抽样**（`head -n 500` + 人工 20 问；**禁全量**）→ ③ 英文 MultiHop-RAG / HotpotQA 子集。不适合：FB15k、Wikidata 全量。原则：**先 10 边通 Neo4j；公开集 ≤50 问 + ≤500 三元组**。

**ES**：中文必须 IK（8.12.2 同版）；无 IK 建过的索引要 `DELETE` 再建。`SPARSE_BACKEND=elasticsearch` 后双写 ES+向量库。

**Neo4j**：问「谁负责/哪个部门/什么关系」时 retrieve **额外** 1～2 跳；边带 `acl`。字段 `h,rel,t,source,acl`。全库 LLM 抽三元组又贵又脏。

**RAGAS**（占位）：Faithfulness / Context Precision；judge=DeepSeek；须注明 LLM-as-judge。hybrid vs graph 消融须同时有图+多跳题。

---

## 9. 仓库结构

```
config/rag.yaml              # 沙盒：引擎与检索阈值（不含密钥）
config/rag.crud.yaml         # CRUD 评测轨（与沙盒隔离）
.env / .env.example          # Key 与连接串
src/
  ingest/                    # Source + ParserRouter
  parsers/                   # Parser / Chunker
  stores/                    # SparseIndex / VectorIndex 插件
  ledger/                    # 入库账本
  llm/                       # 硅基 embed/rerank、生成、限流
  pipeline_hybrid/           # ingest / hybrid / rerank / 分数过滤
  pipeline_graph/            # Neo4j 小图谱（可选）
  agent/                     # LangGraph
  eval/                      # golden.py；crud/ 四任务评测
  app.py                     # Gradio
scripts/                     # ingest / search / eval_* / init_mysql / probe …
alembic/                     # 账本迁移（勿删 versions）
data/sandbox/                # 企业沙盒制度
data/eval/golden.jsonl       # 黄金集
data/eval/org_triples.jsonl  # 沙盒小图谱
data/corpus/80000_docs/      # CRUD 合集（一行一篇）
data/corpus/crud_split/      # CRUD 评测 JSON
docker/                      # ES IK 镜像、Milvus user.yaml 等
```

## 10. 安装



### 10.1 本机要求

- Python ≥ 3.12，推荐 [uv](https://github.com/astral-sh/uv)
- Docker Desktop（Intel Mac 建议内存 ≥ 4GB）
- 硅基流动 API Key、DeepSeek API Key
- MySQL（可用本机已有实例，或 Docker）



### 10.2 依赖

```bash
cp .env.example .env   # 填写 SILICONFLOW_* / DEEPSEEK_* / MYSQL_DSN 等
uv sync
```



### 10.3 基础设施（Docker）

**最快**：只起默认 Qdrant → 见 [11.1](#111-最小路径compose-只起-qdrant)（`docker compose up -d`）。  
ES / Neo4j：`docker compose --profile es|graph up -d`（ES 需 `--build`）。

不想用 compose 时，用下面的 `docker run`。全部绑 `127.0.0.1`，避免 macOS 代理把 `localhost` 转走。

**不要一次全开**：沙盒常见要 **向量库（Qdrant 或 Milvus）+ MySQL 账本**；ES / Neo4j / Postgres / Langfuse 按需。

```bash
docker network create rag-net 2>/dev/null || true
```

若报 `name is already in use`，说明容器已存在，用 `docker start <名字>` 即可。


| 服务                  | 主机端口        | `.env`              | 何时需要                                |
| ------------------- | ----------- | ------------------- | ----------------------------------- |
| Qdrant              | 6333        | `QDRANT_URL`        | `vector.backend=qdrant`             |
| Elasticsearch       | 9200        | `ELASTICSEARCH_URL` | `sparse.backend=elasticsearch`      |
| Kibana              | 5601        | —                   | 看 ES 索引（可选）                         |
| Neo4j               | 7474 / 7687 | `NEO4J_URI`         | 阶段 8 小图谱                            |
| Milvus              | 19530       | `MILVUS_URI`        | `vector.backend=milvus`             |
| MySQL（入库账本）         | 3306        | `MYSQL_DSN`         | `ledger.backend=mysql`（默认）          |
| Postgres + pgvector | 5432        | `POSTGRES_DSN`      | `pgvector` / `postgres_fts`             |
| Langfuse            | 3000        | `LANGFUSE_HOST`     | 自托管 trace；默认请用云                     |


通用注意：

- 本机请用 `127.0.0.1`，避免代理干扰
- `docker compose build` 打的镜像会带 Compose 项目标签，Desktop 会分组；要扁平列表请用 `docker build`
- 与 yaml 中 `sparse.backend` / `vector.backend` / `ledger.backend` **保持一致**后再 ingest



#### 10.3.1 Qdrant

```bash
docker run -d --name qdrant --network rag-net --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:6333
```



#### 10.3.2 Elasticsearch（中文 BM25，建议 IK）+ Kibana

版本必须 **8.12.2 + 同版本 IK**。官方镜像没有 IK，中文召回会差；本仓库用带 IK 的构建：

```bash
docker build -t agentic-rag-es:8.12.2 docker/elasticsearch

docker run -d --name elasticsearch --network rag-net --restart unless-stopped \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e xpack.security.http.ssl.enabled=false \
  -e xpack.ml.enabled=false \
  -e ingest.geoip.downloader.enabled=false \
  -e cluster.routing.allocation.disk.threshold_enabled=false \
  -e 'ES_JAVA_OPTS=-Xms512m -Xmx512m' \
  -p 127.0.0.1:9200:9200 \
  -m 2g --ulimit nofile=65536:65536 \
  -v es_data:/usr/share/elasticsearch/data \
  agentic-rag-es:8.12.2

curl -s http://127.0.0.1:9200
curl -s http://127.0.0.1:9200/_cat/plugins   # 应含 analysis-ik
```

可选 Kibana（与 ES 同为 **8.12.2**）。本仓库 ES **关闭了 security**，不要走 Enrollment token；配好 `ELASTICSEARCH_HOSTS` 即可直连：

```bash
docker rm -f kibana 2>/dev/null || true

docker run -d --name kibana --network rag-net --restart unless-stopped \
  -e ELASTICSEARCH_HOSTS=http://elasticsearch:9200 \
  -e XPACK_SECURITY_ENABLED=false \
  -e XPACK_ENCRYPTEDSAVEDOBJECTS_ENCRYPTIONKEY=abcdefghijklmnopqrstuvwxyzabcdef \
  -p 127.0.0.1:5601:5601 \
  docker.elastic.co/kibana/kibana:8.12.2
```

- 浏览器：[http://127.0.0.1:5601](http://127.0.0.1:5601)（不应再要 Enrollment token）
- Dev Tools：`GET rag_chunks/_search`
- 若仍出现 Enrollment 页：执行 `docker rm -f kibana` 后重建；并确认 ES **没有**设 `xpack.security.enrollment.enabled=true`
- 加密密钥须 ≥32 字符（上面示例刚好 32）；不要把该密钥填进 Enrollment 框

```bash
export SPARSE_BACKEND=elasticsearch
# 或在 config/rag.yaml 设 sparse.backend: elasticsearch
uv run python scripts/ingest.py
```

也可用 compose：`docker compose --profile es up -d --build`（注意 compose build 的镜像标签问题，见上文）。

#### 10.3.3 Neo4j（小图谱，可选）

```bash
docker run -d --name neo4j --network rag-net --restart unless-stopped \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j_data:/data \
  neo4j:5

uv run python scripts/build_graph.py
```

浏览器：[http://127.0.0.1:7474](http://127.0.0.1:7474) ，用户 `neo4j` / 密码 `password`（与 `.env` 一致）。

#### 10.3.4 Milvus（可选向量库）

单容器嵌入 etcd + 本地盘，不必再起 MinIO。

**内存**：Docker Desktop 默认 ~8GB 时，CRUD 全量（约 70 万+ × 1024 维）入库极易把 standalone **OOM kill**。表象是容器仍 `Up`、WebUI 报 `Channel/Segment Not found in DataNode`、日志 `node not match` / `grpc: the client connection is closing`。请把 Docker Desktop → Settings → Resources → **Memory ≥ 12GB**（与 ES/Kibana 同机建议 **16GB**）。入库期可先停 Kibana 省几百 MB。

先写两个小配置再 `docker run`：

```bash
mkdir -p docker/milvus
cat > docker/milvus/embedEtcd.yaml <<'EOF'
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF
# 不要把 user.yaml 截成空文件。仓库里已关闭 compaction 并收紧 flush/query cache，
# 改完必须重启才生效：docker restart milvus-standalone

docker run -d --name milvus-standalone --network rag-net --restart unless-stopped \
  --security-opt seccomp:unconfined \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
  -e COMMON_STORAGETYPE=local \
  -e DEPLOY_MODE=STANDALONE \
  -e GOMEMLIMIT=5GiB \
  -p 127.0.0.1:19530:19530 \
  -p 127.0.0.1:19091:9091 \
  -v milvus_data:/var/lib/milvus \
  -v "$PWD/docker/milvus/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml" \
  -v "$PWD/docker/milvus/user.yaml:/milvus/configs/user.yaml" \
  milvusdb/milvus:v2.5.4 \
  milvus run standalone
```

探活：`curl -s http://127.0.0.1:19091/healthz` 应返回 **`OK`**（容器内 9091 映射到主机 **19091**）。若返回 `Not all components are healthy`，先别入库。  
`.env`：`MILVUS_URI=http://127.0.0.1:19530`。yaml：`vector.backend: milvus`。

#### 10.3.5 MySQL（入库账本，默认）

替代本地 `data/corpus/ingest_manifest.json`。


| 组件        | 路径                                                      |
| --------- | ------------------------------------------------------- |
| ORM 模型    | `src/ledger/models.py`（`KbDocument` / `KbIngestJob`）    |
| Ledger 实现 | `src/ledger/mysql.py`（SQLAlchemy）                       |
| 配置        | `alembic.ini` + `alembic/env.py`                        |
| 迁移脚本      | `alembic/versions/*.py`（**勿删除**；目录为空时 upgrade **不会建表**） |
| 一键初始化     | `scripts/init_mysql.py`                                 |


表含义：

- `kb_document`：每个 source 的 `content_hash` / 切片数 / 状态（`active` / `deleted` / `failed`）
- `kb_ingest_job`：每次 add / update / delete / 失败的流水
- `alembic_version`：当前迁移版本号

本机**已有** MySQL（例如容器名 `mysql`、端口 `3306`）时不必再起容器。没有时：

```bash
docker run -d --name rag-mysql --network rag-net --restart unless-stopped \
  -e MYSQL_ROOT_PASSWORD=password \
  -e MYSQL_DATABASE=rag_hub \
  -e TZ=UTC \
  -p 127.0.0.1:3306:3306 \
  -v rag_mysql_data:/var/lib/mysql \
  mysql:8.0 \
  --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci

# 等就绪后再 init（可看日志：docker logs -f rag-mysql）
MYSQL_DSN=mysql://root:password@127.0.0.1:3306/rag_hub
uv run python scripts/init_mysql.py
```



##### 首次初始化（推荐；本机已有 MySQL 时从这里开始）

```bash
# .env
MYSQL_DSN=mysql://root:你的密码@127.0.0.1:3306/rag_hub

uv sync
uv run python scripts/init_mysql.py
# 作用：CREATE DATABASE IF NOT EXISTS + alembic upgrade head
```



##### 仅升级表结构（库已在、只要升级）

拉代码或改模型并生成迁移后，**不必重建库**：

```bash
uv run alembic upgrade head
```

（再跑一遍 `init_mysql.py` 也可以：建库是 IF NOT EXISTS，随后同样 `upgrade head`。）

##### 开发：改 ORM 后生成新迁移

```bash
# 1. 编辑 src/ledger/models.py
# 2. 生成迁移（务必打开 alembic/versions/ 里新文件检查）
uv run alembic revision --autogenerate -m "describe change"
# 3. 应用到数据库
uv run alembic upgrade head
```



##### 账本相关注意

- **不要**删掉 `alembic/versions/` 下的 `*.py`，否则 `upgrade head` 无事可做，表现为「init 了但没有 kb 表」
- **不要**再手工执行 `sql/mysql/*.sql` 建表；以 Alembic 为准
- 临时退回文件账本：`config/rag.yaml` 里 `ledger.backend: json`，或 `export LEDGER_BACKEND=json`
- 若库里**已有手工建的 kb_* 表**且没有 `alembic_version`：先 `uv run alembic stamp head`，再改表走正常 revision



#### 10.3.6 Postgres + pgvector / FTS（可选）

同一 Postgres 可同时做稀疏（`postgres_fts`）与向量（`pgvector`）。镜像须带 **pgvector** 扩展：

```bash
docker run -d --name rag-postgres --network rag-net --restart unless-stopped \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag \
  -e TZ=UTC \
  -p 127.0.0.1:5432:5432 \
  -v rag_pg_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

until docker exec rag-postgres pg_isready -U postgres; do sleep 1; done
docker exec rag-postgres psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
# 若要自托管 Langfuse，再建库：
docker exec rag-postgres psql -U postgres -c "CREATE DATABASE langfuse;"
```

`.env`：

```bash
POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5432/rag
# 可选表名（插件首次连接时自动建表）
POSTGRES_FTS_TABLE=rag_chunks
PGVECTOR_TABLE=rag_embeddings
```

yaml：

```yaml
sparse:
  backend: postgres_fts
  postgres_fts: { dsn: ${POSTGRES_DSN}, table: rag_chunks }
vector:
  backend: pgvector
  pgvector: { dsn: ${POSTGRES_DSN}, table: rag_embeddings }
```

说明：

- **FTS**：入库用 jieba 切词写入 `to_tsvector('simple', …)`；查询用切词 **OR** `to_tsquery` + `ts_rank`（默认镜像无 zhparser）。ACL 用 `TEXT[]` + `&&` / `'all' = ANY(acl)`。
- **pgvector**：`vector(embed_dim)` + HNSW（`vector_cosine_ops`）；检索 `ORDER BY embedding <=> query`，对外 score = `1 - distance`。维变须**新建表**（或改表名）。
- 探活：`uv run python scripts/probe.py`（当前 yaml 指向上述 backend 时会建表并空搜）。

#### 10.3.7 Langfuse（推荐云；自托管才用 Docker）

日常用 [https://cloud.langfuse.com](https://cloud.langfuse.com)：建项目后把 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 写入 `.env`，`LANGFUSE_HOST` 用云地址。

本机自托管是 **v3 多容器**（需上一节 Postgres 已建 `langfuse` 库）。密钥仅供本地 Demo：

```bash
docker run -d --name langfuse-redis --network rag-net --restart unless-stopped \
  redis:7 redis-server --requirepass myredissecret --maxmemory-policy noeviction

docker run -d --name langfuse-clickhouse --network rag-net --restart unless-stopped \
  -e CLICKHOUSE_DB=default \
  -e CLICKHOUSE_USER=clickhouse \
  -e CLICKHOUSE_PASSWORD=clickhouse \
  -v langfuse_ch_data:/var/lib/clickhouse \
  clickhouse/clickhouse-server:24.8

docker run -d --name langfuse-minio --network rag-net --restart unless-stopped \
  -p 127.0.0.1:9090:9000 \
  -e MINIO_ROOT_USER=minio \
  -e MINIO_ROOT_PASSWORD=miniosecret \
  -v langfuse_minio_data:/data \
  --entrypoint sh minio/minio \
  -c 'mkdir -p /data/langfuse && minio server --address :9000 --console-address :9001 /data'
```

```bash
cat > /tmp/langfuse.env <<'EOF'
DATABASE_URL=postgresql://postgres:postgres@rag-postgres:5432/langfuse
NEXTAUTH_URL=http://127.0.0.1:3000
NEXTAUTH_SECRET=mysecret
SALT=mysalt
ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000
TELEMETRY_ENABLED=false
CLICKHOUSE_MIGRATION_URL=clickhouse://langfuse-clickhouse:9000
CLICKHOUSE_URL=http://langfuse-clickhouse:8123
CLICKHOUSE_USER=clickhouse
CLICKHOUSE_PASSWORD=clickhouse
CLICKHOUSE_CLUSTER_ENABLED=false
REDIS_HOST=langfuse-redis
REDIS_PORT=6379
REDIS_AUTH=myredissecret
LANGFUSE_S3_EVENT_UPLOAD_BUCKET=langfuse
LANGFUSE_S3_EVENT_UPLOAD_REGION=auto
LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=minio
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=miniosecret
LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://langfuse-minio:9000
LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true
LANGFUSE_S3_EVENT_UPLOAD_PREFIX=events/
LANGFUSE_S3_MEDIA_UPLOAD_BUCKET=langfuse
LANGFUSE_S3_MEDIA_UPLOAD_REGION=auto
LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID=minio
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=miniosecret
LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT=http://langfuse-minio:9000
LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE=true
LANGFUSE_S3_MEDIA_UPLOAD_PREFIX=media/
LANGFUSE_INIT_ORG_ID=rag-hub
LANGFUSE_INIT_ORG_NAME=rag-hub
LANGFUSE_INIT_PROJECT_ID=sandbox
LANGFUSE_INIT_PROJECT_NAME=sandbox
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-local-sandbox
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-local-sandbox
LANGFUSE_INIT_USER_EMAIL=admin@local.test
LANGFUSE_INIT_USER_NAME=admin
LANGFUSE_INIT_USER_PASSWORD=Password123!
EOF

docker run -d --name langfuse-worker --network rag-net --restart unless-stopped \
  --env-file /tmp/langfuse.env \
  langfuse/langfuse-worker:3

docker run -d --name langfuse-web --network rag-net --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  --env-file /tmp/langfuse.env \
  langfuse/langfuse:3
```

`.env`：

```bash
LANGFUSE_HOST=http://127.0.0.1:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-sandbox
LANGFUSE_SECRET_KEY=sk-lf-local-sandbox
```

打开 [http://127.0.0.1:3000](http://127.0.0.1:3000) ，`admin@local.test` / `Password123!`。官方说明：[https://langfuse.com/self-hosting/deployment/docker-compose](https://langfuse.com/self-hosting/deployment/docker-compose) 。

#### 10.3.8 停掉 / 清理

```bash
docker stop qdrant elasticsearch kibana neo4j milvus-standalone rag-mysql rag-postgres \
  langfuse-web langfuse-worker langfuse-minio langfuse-redis langfuse-clickhouse
```

数据在 named volume 里，再 `docker start` 即可。删容器：`docker rm <名字>`；清数据：`docker volume rm <卷名>`。

compose 起的服务：`docker compose down`（加 `-v` 会删卷）。

---




## 11. 快速上手



### 11.1 最小路径（compose 只起 Qdrant）

适合：`vector.backend=qdrant`，稀疏用 `memory_jsonl`，账本用本机/已有 MySQL。

```bash
cp .env.example .env          # 填写 SILICONFLOW / DEEPSEEK / MYSQL_DSN
docker compose up -d          # 仅 Qdrant（见 docker-compose.yml）
uv sync
uv run python scripts/init_mysql.py
# 按 config/rag.yaml 探活：硅基 + sparse/vector/ledger（如 es / qdrant / milvus / mysql）
uv run python scripts/probe.py
uv run python scripts/ingest.py
uv run python scripts/search_hybrid.py --search "试用期几个月" --role employee
uv run python scripts/search_hybrid.py --search "一线城市午餐补贴多少" --role employee
uv run python src/app.py
```

同一句「午餐补贴」：`finance` 能答、`employee` 应信息不足。

```bash
uv run python scripts/search_hybrid.py --search "一线城市午餐补贴多少" --role finance
uv run python scripts/eval_golden.py
```

可选增强：

```bash
docker compose --profile es up -d --build    # + Elasticsearch（IK）
docker compose --profile graph up -d         # + Neo4j
uv run python scripts/build_graph.py
```

> 当前仓库 `config/rag.yaml` 若是 `elasticsearch` + `milvus` + `mysql`，需先起对应容器再跑 probe，不能只 `compose up` Qdrant。



### 11.2 已按第 10 节用 docker run 起好服务时

确认 `.env` 与 `config/rag.yaml` 中的向量/稀疏/账本后端已启动且一致，然后：

```bash
uv run python scripts/probe.py   # 失败会点名哪个 backend
uv run python scripts/ingest.py
uv run python scripts/search_hybrid.py --search "试用期几个月" --role employee
uv run python scripts/eval_golden.py
uv run python src/app.py
```

---





## 12. 配置要点

### 12.1 分层原则

| 文件 | 放什么 |
| --- | --- |
| `.env` | 密钥、DSN、URI、`LOG_LEVEL` |
| `config/rag.yaml` / `rag.crud.yaml` | 选哪个插件、切片大小、检索阈值、路由规则 |

### 12.2 入库：Source 与 Router

```yaml
ingest:
  source:
    type: local_glob          # 或 line_corpus（CRUD）
    glob: data/sandbox/**/*.md
  routing:
    - match: {ext: [md, markdown]}
      parser: markdown
      chunker: markdown_heading
  default:
    parser: markdown
    chunker: markdown_heading
```

语料路径写在 **`ingest.source`**，不要写进某个 parser 小节。

### 12.3 检索与完整样例（节选）

```yaml
chunking:
  backend: markdown_heading
  markdown_heading: { chunk_size: 512, chunk_overlap: 64 }

sparse:
  backend: memory_jsonl       # 或 elasticsearch
  # elasticsearch: { url: ${ELASTICSEARCH_URL}, index: rag_chunks }

vector:
  backend: qdrant             # 或 milvus
  # milvus: { uri: ${MILVUS_URI}, collection: ... }

embed_dim: 1024

retrieve:
  recall_k: 10
  top_k: 5
  rrf_k: 60
  sparse_score: { gt: 0 }     # BM25 0 分丢掉
  vector_score: { gte: null } # 沙盒可关；生产可试 ~0.35
  rerank_score: { gte: null } # 主阈值建议放精排，用黄金集扫
```

`retrieve.*_score` 对**各路原始分**过滤，不要拿 BM25 分和向量分直接比。**不要对 RRF 设全局 0.7**。

### 12.4 限流

硅基调用通过 `RateBudget(key, rpm=..., tpm=...)` 传参；`before_call` / `record`，不读环境变量里的 RPM/TPM。

### 12.5 探活

```bash
uv run python scripts/probe.py
# CRUD：uv run python scripts/probe.py --config config/rag.crud.yaml
```

| 组件 | 行为 |
| --- | --- |
| 硅基 | embed 维数 = `embed_dim`；rerank 冒烟 |
| sparse | `memory_jsonl` 查路径；`elasticsearch` ping + IK；`postgres_fts` 建表+空搜；未实现直接失败 |
| vector | `qdrant` / `milvus` 列 collection；`pgvector` 建表+空搜 |
| ledger | `mysql` `SELECT 1`；`json` 查路径 |

### 12.6 环境变量覆盖

```bash
export SPARSE_BACKEND=elasticsearch
export LEDGER_BACKEND=json
uv run python scripts/ingest.py
```

未实现的 backend **启动时报错**，不会静默写错库。

### 12.7 账本迁移（摘要）

| 场景 | 命令 |
| --- | --- |
| 新环境建库+建表 | `uv run python scripts/init_mysql.py` |
| 只升级到最新迁移 | `uv run alembic upgrade head` |
| 改 `models.py` 后出迁移 | `revision --autogenerate` → 检查 → `upgrade head` |

细节见 [10.3.5](#1035-mysql入库账本默认)。

---

## 13. 评估与指标

评测回答两件事：**系统有没有按业务约束工作**（沙盒黄金集），以及**生成质量相对参考答案如何**（CRUD 公开基准）。二者不可互相替代。

### 13.1 为何要评测（用途）

| 目的 | 用什么 | 说明 |
| --- | --- | --- |
| 回归权限 / 拒答 / 出处 | 人写黄金集 | 改切片、ACL、Prompt 后能否守住底线 |
| 对比检索是否增益 | CRUD + `--baseline no_retrieve` | 有检索 vs 空上下文 |
| 对齐论文式生成质量 | CRUD BLEU / ROUGE-L 等 | 与官方任务同构；**勿与论文表直接横比**（嵌入/chunk 不同） |
| 语义相似度辅助 | 可选 bertScore（text2vec） | 非 HuggingFace bert-score |
| 事实覆盖辅助 | 可选 RAGQuestEval | 从 GT 出题，用生成文作答；额外 LLM 成本高 |
| 仪表盘式 faithfulness | RAGAS（占位） | LLM-as-judge，不能替代金标 |

**禁止**：让模型编标准答案再给自己打分当主结论。

### 13.2 沙盒黄金集（企业 ACL 轨）

**数据**：`data/eval/golden.jsonl`。**命令**：`uv run python scripts/eval_golden.py`。  
**产出**：`results/golden_report.json`、`results/badcases.md`。

#### 字段与期望（用途）

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定编号 |
| `query` | 用户原话 |
| `roles` | 用哪个角色去问 |
| `expect` | `answer` / `refuse` / `forbidden` |
| `must_include` | 答案应含关键词（可选） |
| `allowed_sources` | 允许引用的文件名 |

| expect | 业务含义 | 判定意图 |
| --- | --- | --- |
| `answer` | 库中有且角色可见 | 不应拒答；关键词命中；来源 ⊆ allowed |
| `refuse` | 库中确实没有 | 须含「信息不足」 |
| `forbidden` | 库中有但角色不可见 | 须拒答，且不得泄漏机密来源/词 |

#### 流程

```
golden.jsonl
  → 每条 invoke_rag(query, roles)
  → grade：拒答是否正确、关键词、来源是否越权
  → 错题写入 badcases.md（归因见下）
```

#### 归因口径（改哪一层）

| 归因 | 现象 |
| --- | --- |
| 解析/切片 | 表被切错、chunk 无主语 |
| 召回 | 两路都没中 |
| 精排 | 召回里有、top-k 被挤掉 |
| 生成 | 资料对，模型仍编数字 |
| 权限 | 不该中的中了 / 该中的被滤掉 |

建议 ≥20 条，三类期望都有。CRUD 样本**勿**混进此文件。

### 13.3 CRUD-RAG 四任务评测轨

与沙盒隔离：独立 ES index `crud_chunks`、向量 collection `crud_bge_m3`、`tenant_id=crud`；生成用 DeepSeek + `src/eval/crud/prompts/`（**不**走企业拒答 SYSTEM）；`roles: ["all"]`。

#### 任务与检索 query（流程第一步）

| 短名 | split 键 | CRUD 含义 | 检索 query 字段 | GT 字段 | Prompt |
| --- | --- | --- | --- | --- | --- |
| `summary` | `event_summary` | Delete/缩减·摘要 | `event` | `summary` | `summary.txt` |
| `continue` | `continuing_writing` | Create/新增·续写 | `beginning` | `continuing` | `continue_writing.txt` |
| `hallu` | `hallu_modified` | Update/校正 | `newsBeginning` | `hallucinatedMod` | `hallu_mod.txt` |
| `qa`/`qa1` | `questanswer_1doc` | Read/问答 | `questions` | `answers` | `quest_answer.txt` |
| `qa2`/`qa3` | `questanswer_2docs`/`_3docs` | 多文档问答 | 同上 | 同上 | 同上 |

每条样本流程：

```
读 split_merged.json
  → hybrid_search(query, recall_k=16, top_k=8, rerank)
  → 任务 prompt（填 search_documents）→ DeepSeek
  → 解析 <response>（无标签则取全文）
  → 与 GT 算指标 → 追加 preds.jsonl
```

对照位：`--baseline no_retrieve`（空上下文，测「检索是否有用」）。

#### 命令

```bash
uv sync --extra eval
uv run python scripts/probe.py --config config/rag.crud.yaml
uv run python scripts/ingest.py --config config/rag.crud.yaml --limit 1000
# 续跑：--after <账本最新 source>；全量去掉 --limit/--after

uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task all --limit 3
uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task qa --limit 50
uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task qa --limit 50 \
  --baseline no_retrieve --out results/crud_noretrieve
# 可选：--bert-score --quest-eval --sleep 0.5 --no-rerank
```

产出：`results/crud/{summary,continue,hallu,qa*}/preds.jsonl`、`report.json`（含配置快照）、`summary.md`。

评测前 Milvus collection 需 **load**（入库期可 release 省内存）。Docker 内存建议见 §10.3.4。

#### 指标：怎么算、用来干什么

对齐官方 CRUD_RAG（`jieba` 分词；BLEU **默认去掉 brevity penalty**）。

| 指标 | 默认 | 计算要点 | 用途 |
| --- | --- | --- | --- |
| **BLEU-1～4 / bleu-avg** | 开 | n-gram 精确率；官方默认 `bleu/BP`（BP=0 则 0） | 看生成与参考的用词重合；偏短/偏长时与「带 BP 的 BLEU」不可直接比 |
| **ROUGE-L** | 开 | 最长公共子序列 F 值 | 看整体语序与覆盖，摘要/续写常用主看指标之一 |
| **length** | 开 | 生成字符数 | 排查空答、过短/过长 |
| **bertScore** | `--bert-score` | text2vec 余弦（非 HF bert-score）；需本机模型/torch | 语义相近但用词不同时的辅助分 |
| **QA_avg_F1 / QA_recall** | `--quest-eval` | 对 GT 抽关键信息出题 → GT 作答过滤「无法推断」→ 用生成文作答 → token F1 与「非无法推断」比例 | 衡量生成是否盖住参考中的事实点；**额外大量 DeepSeek 调用** |

汇总：各任务 `report.json` 的 `overall`；多任务见 `summary.md` 表。Badcase：按 `rouge-L` 升序从 `preds.jsonl` 人工抽（检索落空 / 切碎 / prompt / 胡编）。

**注意**：官方案例多为 bge-base-zh；本仓库 bge-m3 + 硅基重排 → **分数勿与论文表横比**，只适合本仓库前后消融与回归。

### 13.4 RAGAS / 消融（占位）

`src/eval/ragas_eval.py`、`scripts/run_ablation.py` 仍为占位。先稳定黄金集与 CRUD 主指标，再开 LLM-as-judge 仪表盘。

---

## 14. 换项目检查清单

新客户 / 新知识库时按序改，不必重写检索内核：

1. **语料** `ingest.source` + **routing**（md → 将来 Docling PDF → 合规再 LlamaParse）
2. **ACL** 规则；召回前过滤；两路插件一致
3. **切片**：先标题、表整块、块首章节路径；512/64 只给超长纯文本
4. **黄金集**：业务写 ≥20 条，含拒答和越权
5. **Prompt** 角色口吻与禁区（医疗/法务更严）
6. **模型名**：建议仍 BGE 对；维=`embed_dim`
7. **`sparse.backend` / `vector.backend`**
8. **硅基 TPM/RPM**：ingest batch + 重试
9. **`retrieve.*_score`** 按语料重标定；不对 RRF 设阈
10. **账本**：默认 MySQL；无库才 `ledger.backend=json`

**不改**：RRF 公式、Hit/Chunk、query 前缀仅查询侧、rerank 只打候选、无 hits 不生成、工厂+插件目录。

---

## 15. 常见问题与坑

### 15.1 FAQ（运维）

**Q: Gradio / 本机 HTTP 502？**  
A: 使用 `127.0.0.1`，必要时设置 `NO_PROXY=127.0.0.1,localhost`（`src/app.py` 已尽量绕过本地代理）。

**Q: 维数报错？**  
A: BGE-M3 必须 1024；换嵌入模型要改建向量集合，并改 `embed_dim`。

**Q: init_mysql 跑完仍没有 kb 表？**  
A: 检查 `alembic/versions/` 是否还有 `*.py`；空目录时 upgrade 无事可做。不要删版本文件。

**Q: 已有手工建的 kb 表，再跑迁移会冲突？**  
A: 先 `uv run alembic stamp head`，之后改表再 `revision --autogenerate`。

**Q: 第二次 ingest 仍全量嵌？**  
A: 查账本 hash 是否写入；是否已 `alembic upgrade head`；source 名是否稳定。

**Q: employee 仍能搜到财务制度？**  
A: ACL 必须在 Sparse/Vector 的 `search` 内过滤；检查 chunk 的 `acl` 与角色。

**Q: ES 中文召回差？**  
A: 需要 IK；无插件时确认日志是否回退 `standard`；曾用无 IK 建过索引须 `DELETE` 再建。

**Q: Kibana 要 Enrollment token？**  
A: Demo ES 通常关 security，配 `ELASTICSEARCH_HOSTS` 直连；不要把加密密钥填进 Enrollment 框。见 §10.3.2。

**Q: Milvus search 空？**  
A: collection 需 `load`；写入后可 `flush`；确认 `vector.backend=milvus` 且 `healthz=OK`。

**Q: Milvus 容器 Up，但连不上 / WebUI 报 Channel Not found？**  
A: 多半是 **OOM 重启循环**（`docker events` 可见 `oom`）。Docker Desktop Memory ≥12GB（与 ES 同机建议 16GB）；停 Kibana；`docker restart milvus-standalone`；等 `healthz` 为 `OK` 再 `--after` 续跑。见 `docker/milvus/user.yaml`。

**Q: 未实现的 backend？**  
A: 启动即报错，不会静默写错库——这是刻意行为。

### 15.2 常见坑（原理向）

1. Query 前缀加到文档上或两侧都不加  
2. 两路插件 ACL 不一致 → RRF 阶段泄漏  
3. 每次 ingest 删整个 collection → 无增量，Demo 第三项做不出  
4. Rerank 误用 Chat 接口  
5. Qdrant/Gradio 502：代理劫持 localhost → `127.0.0.1` + `trust_env=False`  
6. 维数≠1024 或本地 BGE 与 API 混空间  
7. 把 CRUD 新闻当内网讲权限  
8. 标准答案让模型生成  
9. Agent 无限循环检索 → 固定边或上限 2  
10. jsonl 当生产存储  
11. hybrid 写死 QdrantClient  
12. 未知 backend 静默回退  
13. 整篇 PDF 512 横切  
14. 按网红榜锁死一家解析器（用最脏 20 页对拍）  
15. KgCLUE/CCKS 全量灌 Neo4j  
16. 图谱边不带 ACL  
17. 对 RRF/BM25 设全局 0.7  

---

## 16. 当前局限与下一步

**未做 / 弱项（须写清边界）**

- SSO、完整审计、水印；扫描件 OCR / 签章 / 复杂版式主路径（三家解析器都会翻车；沙盒先 MD，PDF 默认走 Docling **插件占位**）
- Parser：docling / llamaparse / unstructured；Chunker：pdf_layout / table_row / faq_pair
- Sparse：`qdrant_sparse`（`postgres_fts` / `pgvector` **已实现**）
- 多租户硬隔离、按部门账单；在线 Bad Case 回流；高可用与索引备份
- 公开 KG 全量（KgCLUE 量级）；阶段 8 只抽样或沙盒三元组
- RAGAS / 消融脚本仍占位；CRUD 的 bertScore / QuestEval 默认关（需显式打开）

**建议下一步**

1. 固定生产组合（如 ES + Milvus/Qdrant + MySQL），用黄金集锁行为  
2. 落地 PDF Parser + 表整块切片  
3. 对象存储 / 上传 API 作为新 `IngestSource`  
4. 扩大 CRUD 评测规模并做 Badcase 归因  
5. 再考虑 RAGAS、图谱多跳与权限模型升级  

**执行口诀**：D1 硅基+切片 → D2～3 默认插件 ACL 混合检索 → D4～5 约束生成+LangGraph → D6～7 黄金集+Demo 三动作 → 阶段 8 换 ES / 沙盒三元组；新项目抄 §14。

---

*本文为唯一主文档。账本 SQL 备忘：`sql/mysql/README.md`（以 Alembic 为准）。yaml / 插件状态以源码为准。*
