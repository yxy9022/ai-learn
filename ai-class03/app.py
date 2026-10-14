# RAG（检索增强生成）最小示例：上传 PDF → 切片向量化 → 基于文档内容问答
#
# 整体流程：
#   1. 上传pdf文件，读取 PDF 原文            （PyPDFLoader）
#   2. 切分成小片段              （RecursiveCharacterTextSplitter）
#   3. 片段转成向量存入 Chroma   （OpenAIEmbeddings + Chroma）
#   4. 提问时先检索相关片段，再连同问题一起交给大模型回答（ChatOpenAI）
#
# 运行：uv run streamlit run app.py

import os
from dotenv import load_dotenv  # 从 .env 读取密钥
import streamlit as st  # 网页界面框架

# LangChain 相关组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 对话模型 / 向量模型
from langchain_community.document_loaders import (
    PyPDFLoader,
)  # 加载 PDF，每页一个 Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切分
from langchain_chroma import Chroma  # 本地向量数据库
from langchain_core.messages import HumanMessage, AIMessage  # 统一的用户/AI 消息对象

# 把 .env 里的变量（DEEPSEEK_API_KEY、EMBEDDING_*）加载进 os.environ
load_dotenv()

# 页面基础配置：浏览器标签标题与图标
st.set_page_config(page_title="RAG PDF 问答", page_icon="📚")
st.title("📚 RAG：上传 PDF 然后问它")

# ---------------- 侧边栏：模型配置与文件上传 ----------------
with st.sidebar:
    st.markdown("### 设置")
    # 对话模型密钥：默认取 .env，可在页面上临时覆盖；type="password" 表示输入内容打码
    api_key = st.text_input(
        "DeepSeek API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
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
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    embed_model = st.text_input(
        "Embedding 模型",
        value=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    )

    # 上传控件：只接受 pdf；返回 UploadedFile 对象，未上传时为 None
    uploaded = st.file_uploader("上传 PDF", type="pdf")

    # 清空对话：重置 session_state 中的消息列表并立即重跑脚本
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

    # 清除知识库：丢弃内存中的向量库对象（磁盘上的 data/chroma_db 需手动删除）
    if st.button("🗑️ 清除知识库"):
        st.session_state.pop("vectorstore", None)
        st.rerun()


# ---------------- 模型对象 ----------------
# @st.cache_resource：全局资源缓存，参数不变时复用同一个客户端，
# 避免每次页面刷新（rerun）都重新创建连接
@st.cache_resource
def load_llm(key: str):
    """创建对话模型客户端。DeepSeek 兼容 OpenAI 协议，指定 base_url 即可。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,  # 越大回答越发散，越小越确定
    )


@st.cache_resource
def load_embeddings(key: str, base_url: str, model: str):
    """创建向量模型客户端，把文本转成向量（数组）。"""
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=key,
        openai_api_base=base_url,
        # 默认行为会先用 tiktoken 转成 token id 再发送，第三方兼容接口只接受文本，会报 400
        check_embedding_ctx_length=False,
        # 单批最多送几条文本：LangChain 默认 1000，多数第三方接口限制远小于此
        chunk_size=10,
    )


# 没填密钥时不创建客户端，避免页面一打开就抛异常
llm = load_llm(api_key) if api_key else None
embeddings = (
    load_embeddings(embed_key, embed_base_url, embed_model) if embed_key else None
)

# ---------------- 上传 PDF -> 建立向量库 ----------------
if uploaded is not None and uploaded.name:
    if embeddings is None:
        # 没有向量模型就无法入库，给出明确提示而不是抛 404
        st.error(
            "请先在侧边栏填写 Embedding API Key：DeepSeek 没有 /embeddings 接口，"
            "向量化会返回 404"
        )
    else:
        # 1) 把上传的文件落盘（PyPDFLoader 需要真实文件路径，不接受内存对象）
        os.makedirs("data", exist_ok=True)
        path = f"data/{uploaded.name}"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("读取 + 切分 + 向量化..."):
            # 2) 读取 PDF：每个 Document 是一页，page_content 为文本，metadata 含页码
            docs = PyPDFLoader(path).load()

            # 3) 切分：chunk_size 为每段最大字符数，chunk_overlap 为相邻段重叠字符数，
            #    重叠可避免答案被切断在两个片段中间
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=80
            ).split_documents(docs)

            # 4) 向量化并写入 Chroma：每条片段算一次 embedding，
            #    persist_directory 指定落盘位置，重启页面后仍可检索
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./data/chroma_db",
            )

            # 放进 session_state：同一浏览器会话内可跨 rerun 复用
            st.session_state.vectorstore = vectorstore

        st.success(f"✅ {uploaded.name}：{len(chunks)} 个片段已入库")

# ---------------- 渲染历史对话 ----------------
# session_state 是 Streamlit 的会话级存储，脚本每次 rerun 都会重新执行，靠它保留状态
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# ---------------- 提问与回答（RAG 核心）----------------
# 海象运算符：用户输入非空时赋值给 prompt
if prompt := st.chat_input("基于 PDF 提问..."):
    if not api_key:
        st.error("先填 API Key")
    elif "vectorstore" not in st.session_state:
        st.error("先上传 PDF")
    else:
        # 记录并展示用户问题
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索 + 回答..."):
                # 检索：在向量库中找与问题最相似的 k 个片段
                hits = st.session_state.vectorstore.similarity_search(prompt, k=3)

                # 拼接上下文：把检索到的片段作为“资料”塞进提示词
                context = "\n\n".join(h.page_content for h in hits)

                # 提示词约束模型只依据文档回答，避免幻觉
                sys_prompt = (
                    "你只能根据下面【文档内容】回答。"
                    "如果文档里没有，就说：文档中没有提到。\n\n"
                    f"【文档内容】\n{context}\n\n问题：{prompt}"
                )

                # 调用对话模型生成答案
                resp = llm.invoke([HumanMessage(content=sys_prompt)])
                st.markdown(resp.content)

        # 保存 AI 回复，刷新页面后仍能显示
        st.session_state.messages.append(AIMessage(content=resp.content))
