# AI 应用开发学习

从「命令行调一次大模型」到「基于 PDF 的 RAG 问答」的递进练习。
每个子目录是一个可以独立跑起来的小项目，按顺序学下来即可。

## 学习路线

| 顺序 | 目录 | 主题 | 关键技术点 |
| --- | --- | --- | --- |
| 1 | [ai-class01](./ai-class01) | 命令行对话 | `ChatOpenAI` 调用、`load_dotenv()` 读密钥、venv / uv 环境 |
| 2 | [ai-class02](./ai-class02) | 网页对话 | Streamlit 聊天界面、`session_state` 保存上下文 |
| 3 | [ai-class03](./ai-class03) | RAG（Chroma） | PDF 解析、文本切分、embedding、Chroma 向量库、检索后拼进提示词 |
| 4 | [ai-class04](./ai-class04) | RAG（pgvector） | 把向量库换成 Postgres + pgvector，数据入库可查询、可备份 |
| 5 | [ai-class05](./ai-class05) | RAG + 权限过滤 | 给片段打 `dept` 等元数据、检索时按用户部门 `filter`、模拟多身份登录 |

## 前置准备

- **uv**（推荐）：包管理与运行，`uv run` 会自动创建 `.venv` 并安装依赖，不必手动 activate
- **Python 3.10+**
- **Docker**：`ai-class04` / `ai-class05` 需要跑 pgvector 容器（两者**共用**同一个 `ai-pg` 容器，但用不同集合：`ai_demo_docs` 与 `ai_demo_docs_v2`）
- **API Key**：对话统一用 DeepSeek；**向量化不能用 DeepSeek**（它没有 `/embeddings` 接口），需另备百炼 / OpenAI 等

## 运行方式

每个子目录都是独立的 uv 项目，进入目录后运行即可：

```sh
cd ai-class01 && uv run python main.py          # 命令行
cd ai-class02 && uv run streamlit run app.py    # 网页
cd ai-class03 && uv run streamlit run app.py    # RAG（Chroma）
cd ai-class04 && uv run streamlit run app.py    # RAG（pgvector，需先启动数据库容器）
cd ai-class05 && uv run streamlit run app.py    # RAG + 权限过滤（需数据库容器）
```

完整步骤见各目录下的 `README.md`。

数据库相关的两个项目共用一张物理表 `langchain_pg_embedding`，靠集合名区分，清空数据时务必带上集合条件，避免误删另一个项目的数据。

## 项目约定

- 密钥统一放在各目录的 `.env`，模板见 `.env.example`，`.env` 不提交
- 提交 `pyproject.toml` 与 `uv.lock`（可复现环境），不提交 `.venv`
- 根 `.gitignore` 已覆盖：密钥、虚拟环境、模型权重、向量库目录、`data/` 运行产物、日志、系统文件

## 踩坑速查（各项目通用）

| 现象 | 原因与解决 |
| --- | --- |
| `ModuleNotFoundError: No module named 'langchain_openai'` | 用系统 Python 启动了，必须 `uv run ...` 或先 `source .venv/bin/activate` |
| 向量化报 `404` | DeepSeek 不支持 embeddings，换成百炼 / OpenAI 等向量服务 |
| 向量化报 `400 ... contents is neither str nor list of str` | LangChain 默认先用 tiktoken 转 token id，第三方接口只接受文本，需 `check_embedding_ctx_length=False` |
| 向量化报 `400 ... batch size ... larger than 10` | 单批条数超限，需 `chunk_size=10` |
| 端口 8501 被占用 | Streamlit 会自动顺延到 8502，或加 `--server.port 8502` |
| 换向量模型后检索异常 | 不同模型维度不同，删除向量库（Chroma 目录或 pgvector 集合）后重建 |
| 上传的文档只有某个身份能看到 | 上传时「文档所属部门」下拉框默认是第一项 `hr`，标签写进去就不会变；换部门重传，标 `public` 则所有人可见 |
| pgvector 里查不到自己的数据 | 集合名混淆，或统计 SQL 没限定集合；用 `SELECT name FROM langchain_pg_collection;` 先看再查 |
