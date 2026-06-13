"""
DataMate 入口脚本
─────────────────
支持两种启动模式:
  - Streamlit 界面:  python run.py         (默认)
  - 命令行对话:      python run.py --cli   (调试用)

依赖 uv 管理虚拟环境，启动命令:
  uv run python run.py
"""
import sys
import os

# 将项目根目录加入 Python 路径，确保 src 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli():
    """
    命令行对话模式

    用于快速调试 LLM 连通性，无需启动浏览器。
    输入 'exit' 或 'quit' 退出，Ctrl+C 也可退出。
    """
    from src.llm.deepseek_client import chat

    print("=" * 50)
    print("  数据分析智能助手 (CLI 模式)")
    print("  输入 'exit' 或 'quit' 退出")
    print("=" * 50)

    # 维护对话历史上下文
    history = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        # 退出命令
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if not user_input:
            continue

        print("\n助手: ", end="", flush=True)
        try:
            response = chat(user_input, history)
            print(response)
            # 更新对话历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"[错误] {e}")


def run_streamlit():
    """
    启动 Streamlit Web 界面

    通过 streamlit.web.cli.main() 以编程方式启动，
    无需依赖系统 PATH 中的 streamlit 命令。
    默认监听 8501 端口。
    """
    import streamlit.web.cli as st_cli

    # 拼接 streamlit_app.py 的绝对路径
    app_path = os.path.join(os.path.dirname(__file__), "src", "ui", "streamlit_app.py")
    # 构造 streamlit CLI 参数
    sys.argv = ["streamlit", "run", app_path, "--server.port", "8501"]
    st_cli.main()


# ── 入口 ──
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_streamlit()
