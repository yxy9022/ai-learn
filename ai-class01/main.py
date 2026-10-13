# 导入所需模块
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# 初始化 DeepSeek 模型客户端
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# 打印欢迎信息，提示用户如何退出
print("🤖 你好！输入 quit 退出")

# 循环监听用户输入，直到用户退出
while True:
    # 读取用户输入并去掉首尾空白
    msg = input("你: ").strip()

    # 检查退出命令
    if msg.lower() in ("quit", "exit", "q"):
        break

    # 忽略空输入
    if not msg:
        continue

    # 调用大模型生成回复
    resp = llm.invoke([HumanMessage(content=msg)])

    # 输出模型回复
    print(f"AI: {resp.content}\n")
