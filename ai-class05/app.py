# RAG + 权限过滤：在 ai-class04 的基础上，给每个文档片段打上"部门"等元数据，
# 检索时按当前登录用户的部门过滤，实现"不同人看到不同文档"。
#
# 核心思路：
#   1. 上传 PDF 时给每个 chunk 打元数据：dept（部门）、uploaded_by、source_file
#   2. 检索时用 filter 过滤：只返回"本部门 + public"的片段
#   3. 用户看不到、也检索不到不属于自己的内容
#
# 运行：uv run streamlit run app.py

import os
from dotenv import load_dotenv  # 从 .env 读取密钥与数据库配置
import streamlit as st  # 网页界面框架

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 对话模型 / 向量模型
from langchain_community.document_loaders import PyPDFLoader  # 加载 PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切分
from langchain_postgres import PGVector  # Postgres + pgvector 向量库
from langchain_core.messages import HumanMessage, AIMessage  # 用户 / AI 消息对象
from langchain_core.documents import Document  # 文档对象（本文件暂未直接使用）

# 把 .env 里的变量（DEEPSEEK_API_KEY、DB_*、EMBEDDING_*）加载进 os.environ
load_dotenv()

st.set_page_config(page_title="Step2: RAG + 权限", page_icon="🔐")
st.title("🔐 Step 2: 元数据 + 权限过滤")

# ---------------- 用户登录（模拟）----------------
# 真实项目里这里应换成账号密码 / SSO，把用户信息写进 session
if "user" not in st.session_state:
    st.session_state.user = None

# 未登录：展示三个身份按钮，选完立即 rerun 进入主界面
if st.session_state.user is None:
    st.markdown("### 选择登录身份")
    col1, col2, col3 = st.columns(3)  # 三列布局
    with col1:
        if st.button("👤 张三 (HR)", use_container_width=True):
            # clearance 是权限等级，可用于更细粒度的过滤（当前示例未使用）
            st.session_state.user = {"name": "张三", "dept": "hr", "clearance": 1}
            st.rerun()
    with col2:
        if st.button("👤 李四 (研发)", use_container_width=True):
            st.session_state.user = {
                "name": "李四",
                "dept": "engineering",
                "clearance": 2,
            }
            st.rerun()
    with col3:
        if st.button("👤 王五 (财务)", use_container_width=True):
            st.session_state.user = {"name": "王五", "dept": "finance", "clearance": 1}
            st.rerun()
    # 未登录就到此为止，后面的界面与逻辑不再执行
    st.stop()

# 已登录：显示当前身份，作为后续权限过滤的依据
user = st.session_state.user
st.success(
    f"当前登录：{user['name']} | 部门：{user['dept']} | 权限等级：{user['clearance']}"
)

# 退出登录：清空 session 中的用户信息并 rerun，回到身份选择页
if st.button("🚪 退出登录"):
    st.session_state.user = None
    st.rerun()

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.markdown("### 上传文档")
    # 对话模型密钥：默认取 .env，可在页面上覆盖；type="password" 表示输入打码
    api_key = st.text_input(
        "API Key", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password"
    )

    # 向量模型配置：DeepSeek 只提供对话接口，没有 /embeddings，
    # 所以向量化必须另接一家（百炼 / 火山方舟 / OpenAI 等）
    st.markdown("### Embedding 设置")
    st.caption("DeepSeek 只提供对话模型，向量化需另用一家（百炼 / OpenAI 等）")
    embed_key = st.text_input(
        "Embedding API Key",
        value=os.getenv("EMBEDDING_API_KEY", ""),
        type="password",
    )
    embed_base_url = st.text_input(
        "Embedding Base URL",
        value=os.getenv(
            "EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )
    embed_model = st.text_input(
        "Embedding 模型",
        value=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    )

    uploaded = st.file_uploader("上传 PDF", type="pdf")
    # 上传时由人工指定这份文档属于哪个部门，即写入的元数据 dept
    doc_dept = st.selectbox("文档所属部门", ["hr", "engineering", "finance", "public"])

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()


# ---------------- 模型与向量库 ----------------
# @st.cache_resource：全局资源缓存，参数不变时复用同一个客户端，避免每次 rerun 重建
@st.cache_resource
def load_llm(key: str):
    """创建对话模型客户端，DeepSeek 兼容 OpenAI 协议，指定 base_url 即可。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,  # 越大越发散，越小越确定
    )


@st.cache_resource
def load_embeddings(key: str, base_url: str, model: str):
    """创建向量模型客户端，把文本转成向量。
    DeepSeek 没有 /embeddings 接口，所以向量服务地址与模型名由外部传入。"""
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=key,
        openai_api_base=base_url,
        # 默认行为会先用 tiktoken 转成 token id 再发送，第三方兼容接口只接受文本，会报 400
        check_embedding_ctx_length=False,
        # 单批最多送几条文本：LangChain 默认 1000，多数第三方接口限制远小于此
        chunk_size=10,
    )


# 参数名加下划线前缀 _embeddings：告诉 Streamlit 不用对它做哈希（客户端对象无法哈希）
@st.cache_resource
def load_vectorstore(_embeddings):
    """连接 Postgres 上的 pgvector 向量库，集合名为 ai_demo_docs_v2。"""
    conn = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return PGVector(
        embeddings=_embeddings,
        connection=conn,
        collection_name="ai_demo_docs_v2",
        use_jsonb=True,  # 元数据以 jsonb 存储，才能做灵活的过滤查询
    )


# 没填密钥时不创建客户端，避免页面一打开就抛异常
llm = load_llm(api_key) if api_key else None
embeddings = (
    load_embeddings(embed_key, embed_base_url, embed_model) if embed_key else None
)
vectorstore = load_vectorstore(embeddings) if embeddings else None

# ---------------- 上传 PDF：切块 + 打元数据 + 入库 ----------------
if uploaded is not None and uploaded.name:
    if vectorstore is None:
        st.error(
            "请先在侧边栏填写 Embedding API Key：DeepSeek 没有 /embeddings 接口，"
            "向量化会返回 404"
        )
    else:
        # 1) 落盘：PyPDFLoader 需要真实文件路径，不接受内存对象
        os.makedirs("data", exist_ok=True)
        path = f"data/{uploaded.name}"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("读取 + 切块 + 写入（带部门标签）..."):
            # 2) 读取 PDF 并切分：chunk_size 每段最大字符数，overlap 为相邻段重叠字符数
            docs = PyPDFLoader(path).load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=80
            ).split_documents(docs)

            # ✅ 关键：给每个 chunk 打元数据
            # 这些字段会随向量一起存进数据库，检索时用于过滤
            for chunk in chunks:
                chunk.metadata["dept"] = doc_dept  # 部门：权限过滤的核心字段
                chunk.metadata["uploaded_by"] = user["name"]  # 上传人，便于审计
                chunk.metadata["source_file"] = uploaded.name  # 来源文件名，用于标注出处

            # 3) 向量化并写入 pgvector
            vectorstore.add_documents(chunks)

        st.success(f"✅ {uploaded.name}：{len(chunks)} 个片段已写入（部门={doc_dept}）")

# ---------------- 聊天 ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息：用消息对象类型区分角色
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

if prompt := st.chat_input("提问..."):
    if not api_key:
        st.error("先填 API Key")
    elif vectorstore is None:
        st.error("先在侧边栏填写 Embedding 配置")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索（带权限过滤）..."):
                # ✅ 关键：检索时过滤部门
                # public 文档所有人可见，其他只限本部门
                filter_dict = {
                    "$or": [
                        {"dept": user["dept"]},
                        {"dept": "public"},
                    ]
                }

                # 把向量库包装成检索器：k=3 取最相似的 3 条，filter 先做权限过滤
                # 过滤发生在数据库层面，用户拿不到无权访问的内容
                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k": 3,
                        "filter": filter_dict,
                    }
                )
                docs = retriever.invoke(prompt)

                if not docs:
                    # 过滤后为空：要么文档里没有相关内容，要么没有权限
                    st.markdown("❌ 没有找到相关文档（可能你没有权限访问相关内容）")
                else:
                    # 拼接上下文，交给大模型作答
                    context = "\n\n".join([d.page_content for d in docs])
                    # 汇总命中片段的来源文件与部门，展示给用户核对
                    dept_info = " | ".join(
                        [
                            f"{d.metadata.get('source_file')} (dept={d.metadata.get('dept')})"
                            for d in docs
                        ]
                    )
                    resp = llm.invoke(
                        [
                            HumanMessage(
                                content=f"根据以下内容回答：\n{context}\n\n问题：{prompt}"
                            )
                        ]
                    )
                    st.markdown(resp.content)
                    st.caption(f"📎 来源：{dept_info}")

        # 只有拿到片段、生成了回答才追加进历史
        if docs:
            st.session_state.messages.append(AIMessage(content=resp.content))
