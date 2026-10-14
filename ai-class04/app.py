import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Step1: pgvector RAG", page_icon="🐘")
st.title("🐘 Step 1: pgvector 替代 Chroma")

with st.sidebar:
    st.markdown("### 设置")
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
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()
    if st.button("🗑️ 清除知识库"):
        st.session_state.pop("vectorstore", None)
        st.rerun()


# LLM
@st.cache_resource
def load_llm(key: str):
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
    )


# Embeddings
@st.cache_resource
def load_embeddings(key: str, base_url: str, model: str):
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=key,
        openai_api_base=base_url,
        # 默认行为会先用 tiktoken 转成 token id 再发送，第三方兼容接口只接受文本，会报 400
        check_embedding_ctx_length=False,
        # 单批最多送几条文本：LangChain 默认 1000，多数第三方接口限制远小于此
        chunk_size=10,
    )


# PGVector 连接
@st.cache_resource
def load_vectorstore(_embeddings):
    conn = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return PGVector(
        embeddings=_embeddings,
        connection=conn,
        collection_name="ai_demo_docs",
        use_jsonb=True,
    )


llm = load_llm(api_key) if api_key else None
embeddings = (
    load_embeddings(embed_key, embed_base_url, embed_model) if embed_key else None
)
vectorstore = load_vectorstore(embeddings) if embeddings else None

# 上传 PDF
if uploaded is not None and uploaded.name:
    if vectorstore is None:
        st.error(
            "请先在侧边栏填写 Embedding API Key：DeepSeek 没有 /embeddings 接口，"
            "向量化会返回 404"
        )
    else:
        os.makedirs("data", exist_ok=True)
        path = f"data/{uploaded.name}"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("读取 + 切块 + 写入 pgvector..."):
            docs = PyPDFLoader(path).load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=80
            ).split_documents(docs)

            vectorstore.add_documents(chunks)

        st.success(f"✅ {uploaded.name}：{len(chunks)} 个片段已写入 Postgres")

# 聊天
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

if prompt := st.chat_input("提问..."):
    if not api_key:
        st.error("先填 API Key")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索 + 回答..."):
                docs = vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                resp = llm.invoke(
                    [
                        HumanMessage(
                            content=f"根据以下内容回答：\n{context}\n\n问题：{prompt}"
                        )
                    ]
                )
                st.markdown(resp.content)

        st.session_state.messages.append(AIMessage(content=resp.content))
