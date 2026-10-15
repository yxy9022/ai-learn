# RAG + 权限过滤 + 混合检索：在 ai-class05 的基础上增加一个 BM25 关键词检索器，
# 并把两路结果用 RRF 融合后再交给大模型。
#
# 检索为什么要用两路：
#   1. 向量检索：按语义相似度，能召回"意思接近"的片段
#   2. BM25 检索：按关键词匹配，擅长命中专有名词、编号、精确短语
#   3. RRF 融合：不比较两路分数（量纲不同），只按排名倒数相加，取最终 top-k
#
# 注意：BM25 索引建在内存里，语料从 pgvector 读出，且必须和向量检索做同样的部门过滤。
#
# 运行：uv run streamlit run app.py

import os
from dotenv import load_dotenv  # 从 .env 读取密钥与数据库配置
import streamlit as st  # 网页界面框架
import numpy as np
import jieba  # 中文分词：BM25 按词统计，中文不能用空格切

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 对话模型 / 向量模型
from langchain_community.document_loaders import PyPDFLoader  # 加载 PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切分
from langchain_postgres import PGVector  # Postgres + pgvector 向量库
from langchain_core.messages import HumanMessage, AIMessage  # 用户 / AI 消息对象
from langchain_core.documents import Document  # 文档对象
from rank_bm25 import BM25Okapi  # BM25 关键词检索算法

# 把 .env 里的变量（DEEPSEEK_API_KEY、DB_*、EMBEDDING_*）加载进 os.environ
load_dotenv()

# 本项目的集合名（与 ai-class04 的 ai_demo_docs、ai-class05 的 ai_demo_docs_v2 区分）
COLLECTION_NAME = "ai_demo_docs_v3"
RRF_K = 60  # RRF 平滑系数，越大则排名差异的影响越小

st.set_page_config(page_title="Step3: 混合检索", page_icon="🔀")
st.title("🔀 Step 3: 向量 + BM25 混合检索")

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

    st.markdown("### 混合检索")
    use_bm25 = st.checkbox("启用 BM25（关键词）检索", value=True)
    candidate_k = st.slider("每路候选片段数", min_value=5, max_value=50, value=20)
    final_k = st.slider("最终送入模型的片段数", min_value=1, max_value=10, value=5)

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
    """连接 Postgres 上的 pgvector 向量库，集合名为 ai_demo_docs_v3。"""
    conn = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return PGVector(
        embeddings=_embeddings,
        connection=conn,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,  # 元数据以 jsonb 存储，才能做灵活的过滤查询
    )


def bm25_tokenize(text: str) -> list[str]:
    """BM25 按"词"统计，中文没有空格，直接 split() 会把整句当成一个词，
    检索几乎失效。这里用 jieba 分词，再丢掉空白 token。"""
    return [t for t in jieba.cut(text) if t.strip()]


def build_bm25_index(dept: str):
    """读出当前用户有权访问的片段，构建内存里的 BM25 索引。

    两个容易踩的坑：
    1. 列名是 document / cmetadata，集合表主键是 uuid（不是 id）
    2. 必须按 dept 过滤，否则 BM25 会把别的部门内容召回，等于绕过权限
    """
    import psycopg

    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                  AND (e.cmetadata->>'dept' = %s OR e.cmetadata->>'dept' = 'public')
                """,
                (COLLECTION_NAME, dept),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None, []

    docs = [Document(page_content=r[0], metadata=dict(r[1] or {})) for r in rows]
    corpus = [bm25_tokenize(d.page_content) for d in docs]
    return BM25Okapi(corpus), docs


def get_bm25(dept: str):
    """按部门缓存 BM25 索引；换身份或上传新文档后重建。"""
    cached = st.session_state.get("bm25_cache")
    if cached and cached["dept"] == dept:
        return cached["index"], cached["docs"]

    index, docs = build_bm25_index(dept)
    st.session_state["bm25_cache"] = {"dept": dept, "index": index, "docs": docs}
    return index, docs


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
                chunk.metadata["source_file"] = (
                    uploaded.name
                )  # 来源文件名，用于标注出处

            # 3) 向量化并写入 pgvector
            vectorstore.add_documents(chunks)

            # 语料变了，BM25 索引作废，下次检索时重建
            st.session_state.pop("bm25_cache", None)

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
    resp = None
    if not api_key:
        st.error("先填 API Key")
    elif vectorstore is None:
        st.error("先在侧边栏填写 Embedding 配置")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("混合检索（向量 + BM25）..."):
                # ---- 1. 向量检索：语义相近 ----
                filter_dict = {
                    "$or": [
                        {"dept": user["dept"]},
                        {"dept": "public"},
                    ]
                }
                vector_docs = vectorstore.similarity_search(
                    prompt, k=candidate_k, filter=filter_dict
                )

                # ---- 2. BM25 检索：关键词精确匹配 ----
                bm25_docs = []
                if use_bm25:
                    index, pool = get_bm25(user["dept"])
                    if index is not None:
                        scores = index.get_scores(bm25_tokenize(prompt))
                        # 分数为 0 表示完全没命中关键词，直接丢掉
                        top_idx = np.argsort(scores)[::-1][:candidate_k]
                        bm25_docs = [pool[i] for i in top_idx if scores[i] > 0]

                # ---- 3. RRF 融合 ----
                # Reciprocal Rank Fusion：score = Σ 1/(K + rank)
                # 两路分数（余弦距离 vs BM25 分值）量纲不同、不可直接比较，
                # RRF 只用排名，天然解决这个problem；同一片段被两路命中会累加。
                fused = {}  # key -> (融合分数, 命中来源)
                pool_docs = {}  # key -> Document

                def doc_key(d):
                    # 片段去重键：来源文件 + 页码 + 正文开头
                    return (
                        d.metadata.get("source_file"),
                        d.metadata.get("page"),
                        d.page_content[:80],
                    )

                for rank, d in enumerate(vector_docs, start=1):
                    k = doc_key(d)
                    score, tags = fused.get(k, (0.0, set()))
                    fused[k] = (score + 1.0 / (RRF_K + rank), tags | {"向量"})
                    pool_docs.setdefault(k, d)

                for rank, d in enumerate(bm25_docs, start=1):
                    k = doc_key(d)
                    score, tags = fused.get(k, (0.0, set()))
                    fused[k] = (score + 1.0 / (RRF_K + rank), tags | {"BM25"})
                    pool_docs.setdefault(k, d)

                ranked = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[
                    :final_k
                ]
                final_docs = [pool_docs[k] for k, _ in ranked]
                final_tags = ["+".join(sorted(tags)) for _, (_, tags) in ranked]

                if not final_docs:
                    st.markdown("❌ 没有找到相关文档（可能你没有权限访问相关内容）")
                else:
                    context = "\n\n".join(d.page_content for d in final_docs)
                    source_info = " | ".join(
                        f"{d.metadata.get('source_file', '?')} "
                        f"(dept={d.metadata.get('dept', '?')}, {tag})"
                        for d, tag in zip(final_docs, final_tags)
                    )
                    resp = llm.invoke(
                        [
                            HumanMessage(
                                content=f"根据以下内容回答：\n{context}\n\n问题：{prompt}"
                            )
                        ]
                    )
                    st.markdown(resp.content)
                    st.caption(f"📎 来源：{source_info}")
                    st.caption(
                        f"🔍 向量 {len(vector_docs)} 条 / BM25 {len(bm25_docs)} 条 "
                        f"→ 融合后取 {len(final_docs)} 条"
                    )

        if resp:
            st.session_state.messages.append(AIMessage(content=resp.content))
