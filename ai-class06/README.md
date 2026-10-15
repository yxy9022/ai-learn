# 🔀 RAG 问答：BM25 + 混合检索（ai-class06）

在 `ai-class05`（元数据 + 权限过滤）的基础上再加一路 **BM25 关键词检索**，用 **RRF** 与向量检索结果融合后再交给大模型。

同一个问题会走两路：向量负责「语义相近」，BM25 负责「字面精确匹配」。

## 目录

- [为什么需要混合检索](#为什么需要混合检索)
- [核心思路](#核心思路)
- [可调参数](#可调参数)
- [环境准备](#环境准备)
- [项目初始化与运行](#项目初始化与运行)
- [环境变量配置](#环境变量配置)
- [使用步骤](#使用步骤)
- [验证混合检索是否生效](#验证混合检索是否生效)
- [常见问题](#常见问题)
- [与 ai-class05 的差异](#与-ai-class05-的差异)

## 为什么需要混合检索

纯向量（语义相似度）有两个明显弱点：

| 场景 | 纯向量的问题 |
| --- | --- |
| 搜 `ERROR 503`、`VPN-001` 这类编号 / 错误码 | 向量觉得「ERROR 503」和「服务器错误」语义接近，但精确匹配其实更准 |
| 搜产品名 `DeepSeek-V3`、`iPhone 16 Pro` | 向量会把「类似的产品描述」排前面，反而漏掉精确提到名字的段落 |

BM25（关键词检索）擅长精确匹配，向量擅长语义理解，两者结合就是混合检索（Hybrid Search），生产环境基本都是这个组合。

## 核心思路

```
用户提问
├─→ 向量检索 top-20（语义）
├─→ BM25 检索 top-20（关键词）
└─→ RRF 融合（Reciprocal Rank Fusion）合并去重
     → 最终 top-5 喂给 LLM
```

两路结果**不能直接比分数**：一个是余弦距离，一个是 BM25 分值，量纲完全不同。RRF 只用排名：

```python
score = Σ 1 / (K + rank)      # K = 60，rank 从 1 开始
```

关键代码：

```python
for rank, d in enumerate(vector_docs, start=1):   # 向量路
    fused[key] = (score + 1.0 / (RRF_K + rank), tags | {"向量"})

for rank, d in enumerate(bm25_docs, start=1):     # BM25 路
    fused[key] = (score + 1.0 / (RRF_K + rank), tags | {"BM25"})

ranked = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[:final_k]
```

同一片段被两路同时命中，分数会累加，排名自然靠前。

另外两点实现细节：

1. **中文必须分词**：BM25 按「词」统计，直接 `split()` 会把整句当成一个词，检索几乎失效，所以用 `jieba` 分词
2. **BM25 也要做权限过滤**：语料从 Postgres 读出时就带上部门条件，否则会把其他部门的内容召回，等于绕过 `ai-class05` 建好的权限

## 可调参数

侧边栏「混合检索」区：

| 控件 | 默认值 | 作用 |
| --- | --- | --- |
| 启用 BM25（关键词）检索 | 开启 | 关掉就退化成纯向量检索，方便对比效果 |
| 每路候选片段数 | 20 | 向量路与 BM25 路各自先取多少条参与融合 |
| 最终送入模型的片段数 | 5 | RRF 融合后取前几条拼进提示词 |

另有常量 `RRF_K = 60`（越大，排名差异的影响越小）。

## 环境准备

直接复用 `ai-class04` / `ai-class05` 的 pgvector 容器即可：

```sh
docker ps  # 确认 ai-pg 在运行
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

本项目的集合名是 **`ai_demo_docs_v3`**，与 class04 的 `ai_demo_docs`、class05 的 `ai_demo_docs_v2` 分开，互不干扰。

## 项目初始化与运行

```sh
mkdir ai-class06 && cd ai-class06

uv init --no-package
uv add streamlit langchain-openai langchain-community langchain-postgres psycopg psycopg-binary python-dotenv pypdf rank-bm25 jieba

uv run streamlit run app.py
```

浏览器打开 http://localhost:8501。首次运行会看到 jieba 的词典加载日志，约 0.6 秒，属正常现象。

## 环境变量配置

与 `ai-class05` 完全一致，复制 `.env.example` 为 `.env` 并填写：

| 变量 | 说明 |
| --- | --- |
| `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME` | Postgres 连接信息，与容器一致（端口 5433） |
| `DEEPSEEK_API_KEY` | 对话模型密钥 |
| `EMBEDDING_API_KEY` | 向量模型密钥（DeepSeek **不提供**向量接口，需另用一家） |
| `EMBEDDING_BASE_URL` | 向量服务地址，如百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL` | 向量模型名，如 `text-embedding-v3` |

## 使用步骤

1. 选择登录身份（张三 / 李四 / 王五）
2. 侧边栏填写 DeepSeek API Key 与 Embedding 配置
3. 上传 PDF，**注意选择「文档所属部门」**（下拉框默认第一项 `hr`）
4. 在「混合检索」区调整候选数、返回数，或关掉 BM25 做对比
5. 提问，回答下方会显示：
   - `📎 来源`：命中片段的文件名、部门，以及它来自 `向量` / `BM25` / `向量+BM25`
   - `🔍 向量 N 条 / BM25 M 条 → 融合后取 K 条`

想体验 BM25 的价值：搜文档里的专有名词或编号（如「V3.2」「住宿标准」），对比开关 BM25 时的召回差异。

## 验证混合检索是否生效

```sh
# 库里有哪些集合
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT name FROM langchain_pg_collection;"

# 按部门统计本集合的片段（限定集合，避免把其他集合算进去）
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "
SELECT e.cmetadata->>'dept' AS dept,
       e.cmetadata->>'source_file' AS file,
       COUNT(*) AS chunks
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'ai_demo_docs_v3'
GROUP BY 1, 2 ORDER BY 1;"
```

页面侧边栏把「启用 BM25」关掉再问同一个问题，观察来源里是否还出现「BM25」标记，即可确认两路是否都在参与。

## 常见问题

| 现象 | 原因与解决 |
| --- | --- |
| `psycopg.errors.UndefinedColumn: column col.id does not exist` | 集合表主键是 **`uuid`** 不是 `id`；另外数据表的列名是 `document` / **`cmetadata`**，不是 `content` / `metadata` |
| BM25 搜中文没效果 | 用 `split()` 按空格切词对中文无效，必须用 `jieba`（已集成在 `bm25_tokenize()`） |
| 上传新 PDF 后 BM25 搜不到新内容 | 索引缓存未失效；代码里上传后会 `st.session_state.pop("bm25_cache", None)`，若自己改动请注意同步 |
| 没权限的部门内容被 BM25 召回 | BM25 语料查询必须带上部门条件，与向量检索口径保持一致 |
| 切换身份后检索结果没变 | BM25 索引按部门缓存，切身份会重建；若仍异常，刷新页面重试 |
| 集合名不一致，写入的和检索的不是同一个 | 统一用常量 `COLLECTION_NAME = "ai_demo_docs_v3"`，避免两处写不同的名字 |
| 向量化报 `404` / 两个 `400` | 同前几课：换第三方 embedding、`check_embedding_ctx_length=False`、`chunk_size=10` |
| 未填 API Key 时点提问报 `NameError: resp` | 变量未初始化，已在分支开头加 `resp = None` |

## 目录结构

```
ai-class06/
├── .env                 # 密钥与数据库配置（不提交）
├── .env.example         # 配置模板
├── pyproject.toml       # 依赖声明（提交）
├── uv.lock              # 锁定版本（提交）
├── app.py               # 应用源码
├── data/                # 上传的 PDF（不提交）
└── .venv/               # 虚拟环境（不提交）
```

## 与 ai-class05 的差异

| 项目 | ai-class05 | ai-class06 |
| --- | --- | --- |
| 集合名 | `ai_demo_docs_v2` | `ai_demo_docs_v3` |
| 检索方式 | 纯向量 `as_retriever(k=3, filter=...)` | 向量 + BM25，两路各取候选后 RRF 融合 |
| 新增依赖 | — | `rank-bm25`、`jieba`、`numpy` |
| 索引位置 | 全部在 Postgres | Postgres（向量）+ 内存 BM25 索引 |
| 侧边栏 | 密钥、部门 | 额外增加：启用 BM25、候选数、返回数 |
| 回答展示 | 来源文件 + 部门 | 来源文件 + 部门 + 命中路径（`向量` / `BM25` / `向量+BM25`） |

其余流程（登录、打元数据、切块、向量化、入库、生成）与 `ai-class05` 完全一致。

> 提示：BM25 索引建在内存里，每次切换身份或上传文档都会全表重读重建；片段量到几千条以上时，建议改为上传时增量更新或引入 Elasticsearch / OpenSearch。
