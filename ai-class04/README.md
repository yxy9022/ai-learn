# 🐘 RAG PDF 问答：用 pgvector 替代 Chroma（ai-class04）

把 `ai-class03` 的本地向量库 Chroma 换成 PostgreSQL + pgvector：
**换完，向量数据就存在 Postgres 里了，不再是 Chroma 那个本地文件夹。**

技术栈：`Streamlit` + `LangChain` + `PGVector(pgvector)` + `DeepSeek` 对话 + 第三方 Embedding。

## 目录

- [为什么用 pgvector](#为什么用-pgvector)
- [环境准备](#环境准备)
- [项目初始化与运行](#项目初始化与运行)
- [环境变量配置](#环境变量配置)
- [使用步骤](#使用步骤)
- [验证数据是否入库](#验证数据是否入库)
- [常见问题](#常见问题)
- [目录结构](#目录结构)

## 为什么用 pgvector？

Chroma：快，但孤立。pgvector：慢一拍上手，但跟你的业务长在一起，不添乱。

1. **核心原因：少一个系统，数据和向量在同一事务里**
   你本来就有 PostgreSQL 数据库，现在为了向量检索又搞一个 Chroma：多一台服务要部署、多一套要备份、多一套要监控、多一套要升级。
   pgvector 就是一个插件，装上去就能用，不增加任何新东西。
2. **数据和向量容易对不上**
   你往 Chroma 存向量，往数据库存正文。万一：正文写成功了向量写失败、正文更新了向量忘了更新、正文删了向量还在——用户搜出来一段已经删掉的内容，尴尬。
   pgvector 让向量和正文在同一张表、同一条记录、同一个事务里，要么一起成功，要么一起失败。
3. **搜索时要带条件，Chroma 搞不定**
   真实场景你不会只搜"最像的句子"，你会想：只搜这个用户能看的文档、只搜最近 30 天的、只搜已发布未下架的。
   Chroma 的过滤能力很弱，复杂条件基本要查完再回业务库过滤，又慢又麻烦。
4. **备份 / 权限 / 监控全复用**
   你已经配好了 PostgreSQL 的备份策略、权限管理、审计日志、监控告警；换 Chroma？这些全得重新搞一遍。

## 环境准备

### 1. 用 pgvector 镜像创建容器

```sh
docker run -d \
  --name ai-pg \
  -e POSTGRES_USER=ai_user \
  -e POSTGRES_PASSWORD=ai_pass \
  -e POSTGRES_DB=ai_kb \
  -p 5433:5432 \
  pgvector/pgvector:pg16
```

> `-p 5433:5432` 意思是：宿主机 5433 → 容器内 5432。本地 5432 已被占用，所以用 5433。

### 2. 开启向量扩展并验证

```sh
# 开扩展：返回 CREATE EXTENSION，没有报错即可
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 验证扩展存在：能查到一行就说明成功
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# 验证向量表能正常写入。能查出来就说明：pgvector 扩展 + 向量类型 + 读写 全链路 OK。
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "
CREATE TABLE IF NOT EXISTS test_vec (
  id serial PRIMARY KEY,
  content text,
  embedding vector(3)
);
INSERT INTO test_vec (content, embedding) VALUES ('hello', '[1,2,3]');
SELECT content, embedding FROM test_vec;
"
```

## 项目初始化与运行

```sh
mkdir ai-class04 && cd ai-class04

# 生成 pyproject.toml（--no-package 表示这只是个脚本项目）
uv init --no-package

# 添加依赖
uv add streamlit langchain-openai langchain-community langchain-postgres psycopg psycopg-binary python-dotenv pypdf rank_bm25

# 运行
uv run streamlit run app.py
```

浏览器打开 http://localhost:8501。

## 环境变量配置

复制 `.env.example` 为 `.env` 并填写：

| 变量                                                          | 说明                                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME` | Postgres 连接信息，与上面容器保持一致（端口 5433）                       |
| `DEEPSEEK_API_KEY`                                            | 对话模型密钥                                                             |
| `EMBEDDING_API_KEY`                                           | 向量模型密钥（DeepSeek **不提供**向量接口，需另用一家）                  |
| `EMBEDDING_BASE_URL`                                          | 向量服务地址，如百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL`                                             | 向量模型名，如 `text-embedding-v3`                                       |

`.env` 已被 `.gitignore` 忽略，不会提交；也可以在页面侧边栏临时填写。

## 使用步骤

1. 侧边栏填写 DeepSeek API Key
2. 侧边栏填写 Embedding 配置（Key / Base URL / 模型名）
3. 上传 PDF，等待「已写入 Postgres」提示
4. 在输入框提问，模型只依据检索到的片段作答

## 验证数据是否入库

```sh
# 向量条目数：数字在涨 = 确实写进 Postgres 了 ✅
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT COUNT(*) FROM langchain_pg_embedding;"

# 看看集合与维度
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT name, cmetadata FROM langchain_pg_collection;"
```

LangChain 会自动建两张表：`langchain_pg_collection`（集合）、`langchain_pg_embedding`（向量与正文）。

## 常见问题

| 现象                                              | 原因与解决                                                                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `404`（向量化时）                                 | DeepSeek 没有 `/embeddings` 接口，向量化必须换成百炼 / OpenAI 等                                                    |
| `400 ... contents is neither str nor list of str` | LangChain 默认先用 tiktoken 转 token id 再发送，第三方接口只接受文本；代码中已设 `check_embedding_ctx_length=False` |
| `400 ... batch size should not be larger than 10` | LangChain 默认单批 1000 条，第三方接口有上限；代码中已设 `chunk_size=10`                                            |
| 连接数据库失败                                    | 检查容器是否在运行 `docker ps`、端口是否为 5433、`DB_*` 是否与容器一致                                              |
| 换向量模型后检索异常                              | 不同模型维度不同，需清空集合后重建：`DELETE FROM langchain_pg_embedding;`                                           |
| `ModuleNotFoundError: xxx`                        | 用了系统 Python 启动，改用 `uv run streamlit run app.py`                                                            |
| 端口 8501 被占用                                  | Streamlit 会顺延到 8502，或加 `--server.port 8502`                                                                  |

## 目录结构

```
ai-class04/
├── .env                 # 密钥与数据库配置（不提交）
├── .env.example         # 配置模板
├── pyproject.toml       # 依赖声明（提交）
├── uv.lock              # 锁定版本（提交）
├── app.py               # 应用源码
├── data/                # 上传的 PDF（不提交）
└── .venv/               # 虚拟环境（不提交）
```

## 与 ai-class03 的差异

| 项目     | ai-class03                          | ai-class04                     |
| -------- | ----------------------------------- | ------------------------------ |
| 向量库   | Chroma（本地目录 `data/chroma_db`） | PGVector（Postgres 表）        |
| 数据落盘 | 本地文件夹                          | 数据库，可备份、可加过滤条件   |
| 初始化   | 无需外部服务                        | 需要先跑 pgvector 容器并开扩展 |

其余流程（读取 → 切块 → 向量化 → 检索 → 生成）完全一致。
