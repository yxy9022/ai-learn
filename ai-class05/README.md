# 🔐 RAG 问答：pgvector + 元数据 + 权限过滤（ai-class05）

在 `ai-class04` 的基础上加一层权限：**上传时给每个片段打上部门标签，检索时只返回当前用户有权访问的内容**。

同样的 PDF、同样的问题，张三（HR）和王五（财务）看到的答案来源完全不同。

## 目录

- [核心思路](#核心思路)
- [模拟账号](#模拟账号)
- [环境准备](#环境准备)
- [项目初始化与运行](#项目初始化与运行)
- [环境变量配置](#环境变量配置)
- [使用步骤](#使用步骤)
- [验证权限是否生效](#验证权限是否生效)
- [常见问题](#常见问题)
- [与 ai-class04 的差异](#与-ai-class04-的差异)

## 核心思路

1. **上传时打元数据**：每个 chunk 写入 `dept`（部门）、`uploaded_by`（上传人）、`source_file`（来源文件）
2. **检索时过滤**：只返回「本部门 + public」的片段
3. **过滤发生在数据库层**：无权访问的片段根本不会被取出来，而不是取出来再丢弃

关键代码：

```python
# 写入：给每个片段打标签
for chunk in chunks:
    chunk.metadata["dept"] = doc_dept          # 权限过滤的核心字段
    chunk.metadata["uploaded_by"] = user["name"]
    chunk.metadata["source_file"] = uploaded.name

# 检索：只查本部门 + 公开的
filter_dict = {"$or": [{"dept": user["dept"]}, {"dept": "public"}]}
retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": filter_dict})
docs = retriever.invoke(prompt)
```

## 模拟账号

真实项目里这里应是账号密码 / SSO，示例用三个按钮切换身份：

| 用户 | 部门 `dept`   | 权限等级 `clearance` |
| ---- | ------------- | -------------------- |
| 张三 | `hr`          | 1                    |
| 李四 | `engineering` | 2                    |
| 王五 | `finance`     | 1                    |

`clearance` 已写入用户信息，可用于更细的「密级」过滤，当前示例只按部门过滤。

## 环境准备

直接复用 `ai-class04` 的 pgvector 容器即可（无需重建）：

```sh
docker ps  # 确认 ai-pg 在运行
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

若容器不存在，按 [ai-class04 的环境准备](../ai-class04/README.md#环境准备) 创建。

本项目的集合名是 **`ai_demo_docs_v2`**，与 class04 的 `ai_demo_docs` 分开，互不干扰。

## 项目初始化与运行

```sh
mkdir ai-class05 && cd ai-class05

uv init --no-package
uv add streamlit langchain-openai langchain-community langchain-postgres psycopg psycopg-binary python-dotenv pypdf

uv run streamlit run app.py
```

浏览器打开 http://localhost:8501。

## 环境变量配置

复制 `.env.example` 为 `.env` 并填写：

| 变量                                                          | 说明                                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME` | Postgres 连接信息，与容器一致（端口 5433）                               |
| `DEEPSEEK_API_KEY`                                            | 对话模型密钥                                                             |
| `EMBEDDING_API_KEY`                                           | 向量模型密钥（DeepSeek **不提供**向量接口，需另用一家）                  |
| `EMBEDDING_BASE_URL`                                          | 向量服务地址，如百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL`                                             | 向量模型名，如 `text-embedding-v3`                                       |

`.env` 已被 `.gitignore` 忽略；也可以在侧边栏临时填写。

## 使用步骤

1. 选择登录身份（张三 / 李四 / 王五）
2. 侧边栏填写 DeepSeek API Key 与 Embedding 配置
3. 上传 PDF，**注意选择「文档所属部门」**（下拉框默认第一项 `hr`）
4. 提问，回答下方会显示命中片段的来源文件与部门

想验证权限效果：先用张三上传一份标为 `hr` 的文档，退出后换成王五提问，会看到「没有找到相关文档」。

## 验证权限是否生效

```sh
# 库里有哪些集合（ai_demo_docs 是 class04 的，ai_demo_docs_v2 是本项目的）
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "SELECT name FROM langchain_pg_collection;"

# 按部门统计本集合的片段（用 collection_id 限定，避免把其他集合一起算进去）
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "
SELECT e.cmetadata->>'dept' AS dept,
       e.cmetadata->>'source_file' AS file,
       COUNT(*) AS chunks
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'ai_demo_docs_v2'
GROUP BY 1, 2 ORDER BY 1;"

# 查看某条片段的完整元数据
docker exec -it ai-pg psql -U ai_user -d ai_kb -c "
SELECT e.cmetadata FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'ai_demo_docs_v2' LIMIT 3;"
```

## 常见问题

| 现象                                                       | 原因与解决                                                                                                                                                                       |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **上传的文档只有 hr 能看**                                 | 「文档所属部门」下拉框默认第一项是 `hr`，上传时没改，元数据就标成了 `dept=hr`。重新选择部门后再上传；已入库的数据标签不会自动变，需 `DELETE FROM langchain_pg_embedding;` 后重传 |
| 检索结果为空                                               | ① 文档部门与当前身份不匹配；② 换身份后要用对应用户的部门重新上传；③ 把部门选成 `public` 则所有人可见                                                                             |
| 向量化报 `404`                                             | DeepSeek 没有 `/embeddings` 接口，需换百炼 / OpenAI 等                                                                                                                           |
| 向量化报 `400 ... contents is neither str nor list of str` | 已设 `check_embedding_ctx_length=False`，直接发送文本                                                                                                                            |
| 向量化报 `400 ... batch size ... larger than 10`           | 已设 `chunk_size=10`，限制单批条数                                                                                                                                               |
| 连不上数据库                                               | 检查 `docker ps` 中 `ai-pg` 是否运行、`DB_PORT` 是否为 5433                                                                                                                      |
| 换向量模型后检索异常                                       | 维度不同，清空集合重建：`DELETE FROM langchain_pg_embedding;`                                                                                                                    |

## 目录结构

```
ai-class05/
├── .env                 # 密钥与数据库配置（不提交）
├── .env.example         # 配置模板
├── pyproject.toml       # 依赖声明（提交）
├── uv.lock              # 锁定版本（提交）
├── app.py               # 应用源码
├── data/                # 上传的 PDF（不提交）
└── .venv/               # 虚拟环境（不提交）
```

## 与 ai-class04 的差异

| 项目     | ai-class04                       | ai-class05                                                       |
| -------- | -------------------------------- | ---------------------------------------------------------------- |
| 集合名   | `ai_demo_docs`                   | `ai_demo_docs_v2`                                                |
| 元数据   | 只有 PyPDFLoader 自带的页码      | 额外写入 `dept` / `uploaded_by` / `source_file`                  |
| 检索     | `similarity_search(prompt, k=3)` | `as_retriever(search_kwargs={"k": 3, "filter": {"$or": [...]}})` |
| 登录     | 无                               | 模拟三个身份，按部门过滤                                         |
| 回答展示 | 只有答案                         | 答案 + 来源文件与部门标注                                        |

其余流程（读取 → 切块 → 向量化 → 入库 → 检索 → 生成）完全一致。
