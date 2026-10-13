# 调用大模型命令行对话

# 命令

```sh
# 创建虚拟环境
python -m venv .venv

# 激活（Windows 用 .venv\Scripts\activate）
source .venv/bin/activate

# 装依赖
pip install -r requirements.txt

# 运行
python main.py
```

# 技巧

每次重新打开客户端，都有输入source .venv/bin/activate很麻烦。
方案1: .vscode/settings.json 配置

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
```

方案2：别管 venv 了，用 uv。环境、依赖、activate 全让 uv 管，VS Code 选不选解释器都无所谓。

```sh
uv init
uv add -r requirements.txt
uv run python main.py
```
