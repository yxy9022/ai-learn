# 导入所需依赖
import json
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 从 .env 文件中加载环境变量（例如 DeepSeek API Key）
load_dotenv()

# 历史记录文件路径，用于在页面刷新后保留聊天内容
CHAT_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "chat_history.json")


def load_chat_history():
    """从本地 JSON 文件恢复聊天记录。"""
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []

    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    messages = []
    for item in data:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def save_chat_history(messages):
    """把聊天记录保存到本地 JSON 文件。"""
    serializable = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in messages
    ]
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


# 设置页面标题和图标
st.set_page_config(page_title="我的 AI 网页助手", page_icon="🤖")
st.title("🤖 我的 AI 网页助手")

# 右侧添加清空按钮，用于重置聊天记录
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("清空", use_container_width=True):
        st.session_state.messages = []
        if os.path.exists(CHAT_HISTORY_PATH):
            os.remove(CHAT_HISTORY_PATH)
        st.rerun()


# 缓存 LLM 实例，避免每次重新创建，提高性能
@st.cache_resource
def load_llm():
    # 初始化 DeepSeek 兼容的大语言模型
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
    )


# 创建全局 LLM 实例
llm = load_llm()

# 初始化聊天历史：优先从本地文件恢复，再保存在 Session State 中
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# 渲染历史聊天记录
for m in st.session_state.messages:
    # 根据消息类型判断展示为用户还是助手消息
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# 监听用户输入框中的消息
if prompt := st.chat_input("说点什么..."):
    # 先将用户消息保存到会话历史，并落盘
    st.session_state.messages.append(HumanMessage(content=prompt))
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用大模型生成回复
    with st.chat_message("assistant"):
        with st.spinner("AI 正在想..."):
            # 将当前全部消息历史传给模型，保留上下文
            resp = llm.invoke(st.session_state.messages)
            st.markdown(resp.content)

    # 将模型回复保存到聊天历史中，并持久化
    st.session_state.messages.append(AIMessage(content=resp.content))
    save_chat_history(st.session_state.messages)
