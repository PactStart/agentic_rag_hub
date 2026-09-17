# Agentic RAG Hub — 企业级知识库实践说明

> 目标：在本仓库上搭一条**可上线的中文企业知识库**流水线——带权限、可增量、可追溯、可回归评测；检索与存储按配置插件化，避免业务代码绑死某一家向量库。

依赖安装、Docker、账本 Alembic 迁移以**本文第 7～9 节**为准；分阶段原理与验收见 [docs/企业RAG复用教程.md](docs/企业RAG复用教程.md)。

---

## 目录

1. [项目定位](#1-项目定位)
2. [原理：企业 RAG 在做什么](#2-原理企业-rag-在做什么)
3. [端到端步骤](#3-端到端步骤)
4. [术语表](#4-术语表)
5. [技术栈与实现状态](#5-技术栈与实现状态)
6. [仓库结构](#6-仓库结构)
7. [安装](#7-安装)
8. [快速上手](#8-快速上手)
9. [配置要点](#9-配置要点)
10. [评估](#10-评估)
11. [常见问题](#11-常见问题)
12. [当前局限与下一步](#12-当前局限与下一步)

---

## 1. 项目定位

### 1.1 要建成什么样的系统


| 能力        | 含义                                     |
| --------- | -------------------------------------- |
| 混合检索      | BM25（专名/数字）+ 稠密向量（同义改写）→ RRF 融合 → 重排精排 |
| 权限（ACL）   | 检索阶段过滤；无权限 ≈ 库中不存在，而不是生成后再删            |
| 增量入库      | 按文档 `content_hash` 判断增/改/删；账本记录每次作业    |
| 约束生成      | 无证据则拒答（「信息不足」）；答案带出处                   |
| 可观测 / 可回归 | Langfuse（可选）+ 人写黄金集；公开 CRUD 评测轨另立      |




### 1.2 Demo RAG vs 企业 RAG


| 问题       | Demo      | 本仓库取向                       |
| -------- | --------- | --------------------------- |
| 谁能看见文档？  | 谁搜都能中     | Chunk 带 `acl[]`，索引内过滤       |
| 制度改了？    | 全量重嵌      | 账本 + hash，只处理变更             |
| 答案从哪来？   | 「很像」的一段   | 引用 source / section / page  |
| 表格被切开？   | 固定 512 字切 | 先结构切片，表整块保留（解析器到位后）         |
| 怎么知道变好了？ | 肉眼看两条     | `data/eval/golden.jsonl` 回归 |


沙盒语料：`data/sandbox/*.md`（假制度，带 frontmatter ACL）。  
公开评测语料：`data/raw/CRUD_RAG/`（见 [TODO_CRUD_RAG.md](TODO_CRUD_RAG.md)，**勿与沙盒黄金集混用**）。

---



## 2. 原理：企业 RAG 在做什么



### 2.1 一句话

RAG 不是「把文档扔进向量库」，而是：

**解析 → 切片 → 赋元数据 → 双写索引 → 鉴权召回 → 融合精排 → 约束生成 → 评测回归。**

### 2.2 检索四段论

1. **表示**：词袋（BM25）或向量（BGE-M3，1024 维）
2. **召回**：两路各取 Top-N，求「别漏」
3. **排序**：RRF 合路 + reranker 精排，求「别乱」
4. **生成**：只根据证据写答案，求「别编」

两路原始分数量纲不同，**不能直接相加**；RRF 只看排名。Rerank 贵，只对召回后的十几条跑。

### 2.3 为何嵌入/重排走硅基流动

本机加载 `BAAI/bge-m3` / `bge-reranker` 成本高。硅基提供同名 HTTP API：算法故事不变，工程上变成「有 Key 就能嵌」。客户端限流见 `src/llm/rate_limit.py`（`RateBudget`：调用前触顶等待，调用后按响应 token 记账）。


| 能力  | 模型 / 接口                                  |
| --- | ---------------------------------------- |
| 嵌入  | `BAAI/bge-m3` → `/v1/embeddings`         |
| 重排  | `BAAI/bge-reranker-v2-m3` → `/v1/rerank` |
| 生成  | DeepSeek 等 OpenAI 兼容 Chat（默认）            |




### 2.4 插件边界

业务只依赖协议，不写死引擎：

- `IngestSource`：发现文件（当前 `local_glob`）
- `Parser` + `Chunker`：由 `ingest.routing` 按扩展名选择
- `SparseIndex` / `VectorIndex`：BM25 侧 / 向量侧
- `IngestLedger`：增量账本（MySQL 或本地 JSON）

换 ES / Milvus / Qdrant = 改 `config/rag.yaml`，不改 `hybrid_search`。

---



## 3. 端到端步骤

```
① 解析        ingest.source 列文件 → Router 选 Parser
② 切片        Chunker（如 markdown_heading）
③ 元数据      source / page / acl / content_hash / chunk_id
④ 双写        SparseIndex.upsert + VectorIndex.upsert + Ledger
⑤ 检索        稀疏 ∥ 向量 → 分数过滤 → RRF → rerank → top_k（ACL 在插件内）
⑥ 生成        仅据资料；否则拒答；列出处
⑦ 编排        LangGraph：rewrite → retrieve → generate
⑧ 评测上线    黄金集 +（可选）Langfuse + Gradio Demo
```

建议推进顺序（与教程阶段对齐）：


| 阶段  | 主题              | 建议产出                           |
| --- | --------------- | ------------------------------ |
| 0   | Key、探活          | `probe.py` 按 yaml 通硅基+稀疏/向量/账本 |
| 1   | 解析切片            | 沙盒 md → Chunk                  |
| 2   | 入库 + ACL + 账本   | 双写 + 增量跳过                      |
| 3   | 混合检索            | `hybrid_search`                |
| 4   | 拒答与出处           | 无权限/无资料不编                      |
| 5   | LangGraph       | `invoke_rag`                   |
| 6   | 黄金集             | `eval_golden.py`               |
| 7   | Demo            | Gradio 切角色                     |
| 8   | ES / 图谱 / RAGAS | 可选增强                           |


---



## 4. 术语表


| 术语           | 含义                                   |
| ------------ | ------------------------------------ |
| Chunk        | 检索与入库的最小文本单元，含正文与元数据                 |
| ACL          | 访问控制标签列表；与用户 `roles` 有交集或含 `all` 才可见 |
| BM25         | 经典词项检索打分；ES / 内存 jieba+rank_bm25 实现  |
| 稠密向量         | 语义嵌入；本仓库固定 1024 维（BGE-M3）            |
| RRF          | Reciprocal Rank Fusion，按排名融合多路召回     |
| Rerank       | 查询-文档成对精排（bge-reranker）              |
| content_hash | 文档内容指纹；未变则跳过重嵌                       |
| Ledger       | 入库账本：`kb_document` / `kb_ingest_job` |
| Source       | 语料发现层（≠ Parser）                      |
| Router       | 按扩展名把文件路由到 Parser/Chunker            |
| 黄金集          | 人写问答与期望（拒答/必含词/角色），用于回归              |
| CRUD-RAG     | 公开中文 RAG 基准（续写/摘要/纠错/问答）；评测轨另配       |


---



## 5. 技术栈与实现状态

以代码为准（`NotImplementedError` = 尚未实现）。

### 5.1 已实现（可跑通主路径）


| 层       | 选项                                                |
| ------- | ------------------------------------------------- |
| 嵌入 / 重排 | 硅基 BGE-M3 / bge-reranker + `RateBudget`           |
| 生成      | DeepSeek（OpenAI SDK）                              |
| 语料源     | `ingest.source.type=local_glob`                   |
| 解析      | `markdown`（frontmatter ACL）                       |
| 切片      | `markdown_heading`                                |
| 稀疏索引    | `memory_jsonl`、`elasticsearch`（无 IK 时回退 standard） |
| 向量索引    | `qdrant`、`milvus`                                 |
| 账本      | `mysql`（SQLAlchemy + Alembic）、`json`              |
| 编排      | LangGraph 三节点                                     |
| 评测      | `src/eval/golden.py`                              |
| UI      | Gradio（`src/app.py`）                              |
| 可观测     | Langfuse Callback（`langfuse.langchain`，密钥走环境变量）   |




### 5.2 配置已预留、代码未完成


| 项                                          | 状态         |
| ------------------------------------------ | ---------- |
| Parser：docling / llamaparse / unstructured | 占位         |
| Chunker：pdf_layout / table_row / faq_pair  | 占位         |
| Sparse：postgres_fts / qdrant_sparse        | 占位         |
| Vector：pgvector                            | 占位         |
| CRUD 四任务评估脚本                               | 见 TODO，未落地 |
| RAGAS / 消融脚本                               | 占位，请先黄金集   |




### 5.3 当前默认配置（以 `config/rag.yaml` 为准）

仓库里常见默认组合会随你本地修改变化；打开 yaml 核对。示例取向：

- `ingest.source.glob: data/sandbox/**/*.md`
- `sparse.backend`: 可为 `elasticsearch` 或沙盒用 `memory_jsonl`
- `vector.backend`: 可为 `milvus` 或 `qdrant`
- `ledger.backend: mysql`
- `embed_dim: 1024`

最小依赖沙盒可用：**Qdrant + MySQL（或 json 账本）+ memory_jsonl**。

---



## 6. 仓库结构

```
config/rag.yaml              # 引擎与检索阈值（不含密钥）
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
  eval/                      # 黄金集等
  app.py                     # Gradio
scripts/                     # ingest / search / eval / init_mysql / probe …

alembic/                     # 账本迁移（勿删 versions）
data/sandbox/                # 企业沙盒制度
data/eval/golden.jsonl       # 黄金集
data/raw/CRUD_RAG/           # 公开评测集（可选）
docs/企业RAG复用教程.md       # 分阶段长教程
TODO_CRUD_RAG.md             # CRUD 评测轨待办
```

---



## 7. 安装



### 7.1 本机要求

- Python ≥ 3.12，推荐 [uv](https://github.com/astral-sh/uv)
- Docker Desktop（Intel Mac 建议内存 ≥ 4GB）
- 硅基流动 API Key、DeepSeek API Key
- MySQL（可用本机已有实例，或 Docker）



### 7.2 依赖

```bash
cp .env.example .env   # 填写 SILICONFLOW_* / DEEPSEEK_* / MYSQL_DSN 等
uv sync
```



### 7.3 基础设施（Docker）

**最快**：只起默认 Qdrant → 见 [8.1](#81-最小路径compose-只起-qdrant)（`docker compose up -d`）。  
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
| Postgres + pgvector | 5432        | `POSTGRES_DSN`      | `pgvector` / `postgres_fts`（插件尚未实现） |
| Langfuse            | 3000        | `LANGFUSE_HOST`     | 自托管 trace；默认请用云                     |


通用注意：

- 本机请用 `127.0.0.1`，避免代理干扰
- `docker compose build` 打的镜像会带 Compose 项目标签，Desktop 会分组；要扁平列表请用 `docker build`
- 与 yaml 中 `sparse.backend` / `vector.backend` / `ledger.backend` **保持一致**后再 ingest



#### 7.3.1 Qdrant

```bash
docker run -d --name qdrant --network rag-net --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:6333
```



#### 7.3.2 Elasticsearch（中文 BM25，建议 IK）+ Kibana

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

#### 7.3.3 Neo4j（小图谱，可选）

```bash
docker run -d --name neo4j --network rag-net --restart unless-stopped \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j_data:/data \
  neo4j:5

uv run python scripts/build_graph.py
```

浏览器：[http://127.0.0.1:7474](http://127.0.0.1:7474) ，用户 `neo4j` / 密码 `password`（与 `.env` 一致）。

#### 7.3.4 Milvus（可选向量库）

单容器嵌入 etcd + 本地盘，不必再起 MinIO。先写两个小配置再 `docker run`：

```bash
mkdir -p docker/milvus
cat > docker/milvus/embedEtcd.yaml <<'EOF'
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF
: > docker/milvus/user.yaml

docker run -d --name milvus-standalone --network rag-net --restart unless-stopped \
  --security-opt seccomp:unconfined \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
  -e COMMON_STORAGETYPE=local \
  -e DEPLOY_MODE=STANDALONE \
  -p 127.0.0.1:19530:19530 \
  -p 127.0.0.1:19091:9091 \
  -v milvus_data:/var/lib/milvus \
  -v "$PWD/docker/milvus/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml" \
  -v "$PWD/docker/milvus/user.yaml:/milvus/configs/user.yaml" \
  milvusdb/milvus:v2.5.4 \
  milvus run standalone
```

探活：`curl -s http://127.0.0.1:19091/healthz`（容器内 9091 映射到主机 **19091**）。  
`.env`：`MILVUS_URI=http://127.0.0.1:19530`。yaml：`vector.backend: milvus`。

#### 7.3.5 MySQL（入库账本，默认）

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



#### 7.3.6 Postgres + pgvector（可选；插件未实现）

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

`.env`：`POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5432/rag`。  
yaml 的 `pgvector` / `postgres_fts` **代码仍为占位**。

#### 7.3.7 Langfuse（推荐云；自托管才用 Docker）

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

#### 7.3.8 停掉 / 清理

```bash
docker stop qdrant elasticsearch kibana neo4j milvus-standalone rag-mysql rag-postgres \
  langfuse-web langfuse-worker langfuse-minio langfuse-redis langfuse-clickhouse
```

数据在 named volume 里，再 `docker start` 即可。删容器：`docker rm <名字>`；清数据：`docker volume rm <卷名>`。

compose 起的服务：`docker compose down`（加 `-v` 会删卷）。

---



## 8. 快速上手



### 8.1 最小路径（compose 只起 Qdrant）

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
uv run python src/app.pym
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



### 8.2 已按第 7 节用 docker run 起好服务时

确认 `.env` 与 `config/rag.yaml` 中的向量/稀疏/账本后端已启动且一致，然后：

```bash
uv run python scripts/probe.py   # 失败会点名哪个 backend
uv run python scripts/ingest.py
uv run python scripts/search_hybrid.py --search "试用期几个月" --role employee
uv run python scripts/eval_golden.py
uv run python src/app.py
```

---



## 9. 配置要点



### 9.1 分层原则


| 文件                | 放什么                  |
| ----------------- | -------------------- |
| `.env`            | 密钥、DSN、URI           |
| `config/rag.yaml` | 选哪个插件、切片大小、检索阈值、路由规则 |




### 9.2 入库：Source 与 Router

```yaml
ingest:
  source:
    type: local_glob
    glob: data/sandbox/**/*.md
  routing:
    - match: {ext: [md, markdown]}
      parser: markdown
      chunker: markdown_heading
  default:
    parser: markdown
    chunker: markdown_heading
```

语料路径写在 `**ingest.source**`，不要写进某个 parser 小节。

### 9.3 检索阈值

`retrieve.*_score` 对**各路原始分**过滤，不要拿 BM25 分和向量分直接比。主阈值优先放在 `rerank_score`，用黄金集扫，不要抄网上的绝对数字。

### 9.4 限流

硅基调用通过 `RateBudget(key, rpm=..., tpm=...)` 传参；`before_call` / `record`，不读环境变量里的 RPM/TPM。

### 9.5 探活

```bash
uv run python scripts/probe.py
```

读取当前 `config/rag.yaml`（及 `SPARSE_BACKEND` / `LEDGER_BACKEND`），依次检查：


| 组件     | 行为                                                                |
| ------ | ----------------------------------------------------------------- |
| 硅基     | embed 维数 = `embed_dim`；rerank 冒烟                                  |
| sparse | `memory_jsonl` 查路径；`elasticsearch` ping + 是否装 IK；未实现 backend 直接失败 |
| vector | `qdrant` / `milvus` 列 collection；`pgvector` 未实现则失败                |
| ledger | `mysql` `SELECT 1`；`json` 查路径                                     |


任一失败非零退出，便于 CI / 上线前自检。

### 9.6 环境变量覆盖与切换后端

除 `.env` / yaml 外，部分后端可用环境变量临时覆盖（便于脚本切换，不必改 yaml）：

```bash
export SPARSE_BACKEND=elasticsearch   # 覆盖 sparse.backend
export LEDGER_BACKEND=json            # 临时改用文件账本
uv run python scripts/ingest.py
uv run python scripts/build_graph.py  # 需 Neo4j 已起
```

**未实现的** `backend` **会在启动时报错**，不会静默写错库。

### 9.7 账本迁移（摘要）


| 场景                 | 命令                                                                      |
| ------------------ | ----------------------------------------------------------------------- |
| 新环境建库+建表           | `uv run python scripts/init_mysql.py`                                   |
| 只升级到最新迁移           | `uv run alembic upgrade head`                                           |
| 改 `models.py` 后出迁移 | `uv run alembic revision --autogenerate -m "..."` → 检查 → `upgrade head` |


细节见 [7.3.5](#735-mysql入库账本默认)。

---



## 10. 评估



### 10.1 沙盒黄金集（已实现）

- 数据：`data/eval/golden.jsonl`
- 命令：`uv run python scripts/eval_golden.py`
- 检查：拒答 / 必含词 / 角色与 `allowed_sources`
- 产出：`results/golden_report.json`、`results/badcases.md`

验收建议：≥20 条，覆盖 answer / refuse / forbidden。

### 10.2 CRUD-RAG 评测轨（规划中）

公开基准含续写 / 摘要 / 纠错 / 多文档问答与 8 万新闻库。与企业 ACL 沙盒是**两条轨**：

- 独立索引 / collection（勿与 sandbox 共用）
- 指标：ROUGE / BLEU 等（非黄金集规则）
- 待办清单：[docs/TODO_CRUD_RAG.md](docs/TODO_CRUD_RAG.md)



### 10.3 RAGAS / 消融

`src/eval/ragas_eval.py`、`scripts/run_ablation.py` 仍为占位；先稳定黄金集再开。

---



## 11. 常见问题

**Q: Gradio / 本机 HTTP 502？**  
A: 使用 `127.0.0.1`，必要时设置 `NO_PROXY=127.0.0.1,localhost`。

**Q: 维数报错？**  
A: BGE-M3 必须 1024；换嵌入模型要改建向量集合，并改 `embed_dim`。

**Q: init_mysql 跑完仍没有 kb 表？**  
A: 检查 `alembic/versions/` 是否还有 `*.py`；空目录时 Alembic 无迁移可执行。不要删版本文件。

**Q: 已有手工建的 kb 表，再跑迁移会冲突？**  
A: 先 `uv run alembic stamp head` 对齐版本，之后改表再 `revision --autogenerate`。

**Q: 第二次 ingest 仍全量嵌？**  
A: 查账本 hash 是否写入；是否已 `alembic upgrade head`；source 名是否稳定。

**Q: employee 仍能搜到财务制度？**  
A: ACL 必须在 Sparse/Vector 的 `search` 内过滤；检查 chunk 的 `acl` 与角色。

**Q: ES 中文召回差？**  
A: 需要 IK；无插件时确认日志是否回退 `standard`，必要时删索引重建。

**Q: Kibana 要 Enrollment token？**  
A: 本仓库 Demo 通常关闭 ES security，应配置 `ELASTICSEARCH_HOSTS` 直连并重建 Kibana；不要把加密密钥填进 Enrollment 框。见上文「7.3.2」。

**Q: Milvus search 空？**  
A: collection 需 `load`；写入后 `flush`；确认 `vector.backend=milvus` 且服务健康检查通。

**Q: 未实现的 backend？**  
A: 启动即报错，不会静默写错库——这是刻意行为。

---



## 12. 当前局限与下一步

**未做 / 弱项**

- SSO、完整审计、扫描件 OCR 主路径
- PDF 解析器（docling 等）未实现
- GraphRAG 默认只用沙盒少量三元组；公开 KG 请抽样（见教程阶段 8）
- CRUD 完整四任务评估未接入
- **pgvector / Postgres FTS** 仅预留 yaml 与 Docker，代码仍是 `NotImplementedError`（**Milvus 向量插件已实现**）

**建议下一步（企业知识库）**

1. 固定一套生产组合（例如 ES + Milvus/Qdrant + MySQL），用黄金集锁行为
2. 落地 PDF Parser + 表整块切片
3. 对象存储 / 上传 API 作为新的 `IngestSource`
4. 按 [docs/TODO_CRUD_RAG.md](docs/TODO_CRUD_RAG.md) 做公开集评测轨（与 ACL 沙盒隔离）
5. 再考虑 RAGAS、图谱多跳与权限模型升级

---



## 相关文档


| 文档                                         | 用途                      |
| ------------------------------------------ | ----------------------- |
| [docs/企业RAG复用教程.md](docs/企业RAG复用教程.md)     | 分阶段原理与验收                |
| [TODO_CRUD_RAG.md](TODO_CRUD_RAG.md)       | CRUD 评测轨清单              |
| [sql/mysql/README.md](sql/mysql/README.md) | 账本 SQL 说明（以 Alembic 为准） |


---

*文档版本与仓库代码同步整理；若 yaml / 插件状态有变，以源码与* `config/rag.yaml` *为准。*