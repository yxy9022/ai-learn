# 使用网页应用调用大模型对话

# ‌Streamlit介绍

Streamlit 是一个开源的 ‌Python 库‌，专门用来快速把数据脚本变成交互式网页应用，核心就是“‌几行纯 Python 代码就能做出带界面的数据工具‌”，完全不用写前端代码。‌‌

# 初始化命令

```sh
# 生成 pyproject.toml（--bare 不生成示例文件）
uv init --bare --python 3.10

# 安装依赖
uv add streamlit langchain-openai python-dotenv
# 做3件事：
# 1. 往pyproject.toml中写入依赖信息
# 2. 生成 uv.lock：把所有包及其间接依赖锁定到精确版本，保证别人装出一模一样的环境。
# 3. 安装到 .venv：真正下载安装，所以 uv run 能找到它们。
```

# 运行命令

```sh
# 使用uv运行
uv run streamlit run app.py

```

# 其它

**pycache**/ 是 Python 自动生成的文件夹，你每次运行 python app.py 时，Python 内部会做一件事：
把你的 .py 源代码 编译成字节码（.pyc 文件），方便下次更快加载。
这些 .pyc 文件就存在 **pycache**/ 里。

```text
**pycache**/
└── config.cpython-310.pyc
```

cpython-310 表示你用的是 CPython 3.10 版本。
