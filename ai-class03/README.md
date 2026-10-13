# 📚 RAG PDF 问答（ai-class03）

上传一份 PDF，然后像聊天一样基于文档内容提问——一个最小可运行的 RAG（检索增强生成）示例。

技术栈：`Streamlit`（界面） + `LangChain`（编排） + `Chroma`（本地向量库） + `DeepSeek`（对话模型）。

## 功能

- 上传 PDF，自动完成「解析 → 切分 → 向量化 → 入库」
- 提问时先检索最相关的片段，再交给大模型作答
- 密钥既可在侧边栏临时填写，也可写在 `.env` 里

## 环境要求

- `uv`（包管理与运行，会自动准备 Python 3.10+）
- 两个 API Key：对话用 DeepSeek，向量化需另用一家（见下方说明）

## 快速开始

```sh
# 1. 初始化项目（已有 pyproject.toml 可跳过）
uv init --bare

# 2. 安装依赖
uv add streamlit langchain-openai langchain-community langchain-chroma pypdf python-dotenv

# 3. 启动
uv run streamlit run app.py
```

浏览器会自动打开 http://localhost:8501（若端口被占用，加 `--server.port 8502`）。

## 配置密钥

复制 `.env.example` 为 `.env` 并填入：

| 变量                 | 说明                                                                     |
| -------------------- | ------------------------------------------------------------------------ |
| `DEEPSEEK_API_KEY`   | 对话模型密钥，DeepSeek 控制台申请                                        |
| `EMBEDDING_API_KEY`  | 向量模型密钥，DeepSeek **不提供**向量接口，需另用一家                    |
| `EMBEDDING_BASE_URL` | 向量服务地址，如百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL`    | 向量模型名，如百炼 `text-embedding-v3`、OpenAI `text-embedding-3-small`  |

`.env` 已被 `.gitignore` 忽略，不会提交。

## 使用步骤

1. 侧边栏填写 DeepSeek API Key
2. 侧边栏填写 Embedding 配置（Key / Base URL / 模型名）
3. 上传 PDF，等待「已入库」提示
4. 在输入框提问

## 实现原理

一句话：

```
PDF → 切块 → embedding → Chroma → 提问时检索 top-k → 拼进 prompt → LLM 回答
```

对应 `app.py` 中的位置：

| 步骤              | 代码                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------- |
| 对话 / 向量客户端 | `load_llm()`、`load_embeddings()`（`@st.cache_resource` 缓存复用）                     |
| 读取 PDF          | `PyPDFLoader(path).load()`，每页一个 Document                                          |
| 切分              | `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)`，重叠避免答案被切断 |
| 入库              | `Chroma.from_documents(..., persist_directory="./data/chroma_db")`                     |
| 检索              | `similarity_search(prompt, k=3)`                                                       |
| 生成              | 把检索片段拼进提示词，要求模型只依据文档回答                                           |

## 目录结构

```
ai-class03/
├── .env                 # 密钥（不提交）
├── .env.example         # 密钥模板
├── pyproject.toml       # 依赖声明（提交）
├── uv.lock              # 锁定版本（提交）
├── app.py               # 应用源码
├── docs/                # 示例 PDF
├── data/                # 运行时产物：上传的 PDF + chroma_db（不提交）
└── .venv/               # 虚拟环境（不提交）
```

## 常见问题

| 现象                                              | 原因与解决                                                                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `404 NotFoundError`（向量化时）                   | DeepSeek 没有 `/embeddings` 接口，必须换一家向量服务                                                                |
| `400 ... contents is neither str nor list of str` | LangChain 默认先用 tiktoken 转 token id 再发送，第三方接口只接受文本；代码中已设 `check_embedding_ctx_length=False` |
| 端口 8501 被占用                                  | Streamlit 会顺延到 8502；或 `lsof -nP -iTCP:8501 -sTCP:LISTEN` 查到 PID 后 `kill`                                   |
| 换向量模型后检索异常                              | 不同模型维度不同，删除 `data/chroma_db` 后重新上传                                                                  |
| 刷新页面后不能提问                                | `vectorstore` 存在会话状态里，重新上传一次 PDF 即可（磁盘上的库仍在）                                               |

## 示例问题

- 这篇文档主要讲了什么？
- 人工智能是什么时候开始的？
- 文档里提到了哪些关键技术？
