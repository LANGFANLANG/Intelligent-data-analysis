"""
Streamlit 前端主界面
────────────────────
DataMate 智能数据分析助手的核心 UI 页面。

布局结构:
  侧边栏 (sidebar):
    ├── 品牌标识
    ├── 会话管理（新建/切换/删除/重命名）
    ├── 数据加载（上传文件 / 数据预览）
    └── 设置（Temperature 滑块 / 清空对话）

  主区域 (main):
    ├── 标题 + 数据状态指示器
    ├── 会话 ID 显示
    ├── 引导提示（无数据时）
    ├── 聊天消息历史
    └── 输入框

会话持久化:
  - 启动时自动建表 + 恢复最新会话
  - 每轮对话后自动 INSERT 消息到 PostgreSQL
  - 切换会话时自动存档当前会话
"""
import streamlit as st
import pandas as pd
import os

from src.config import DEEPSEEK_MODEL
from src.agent.context import ToolContext
from src.agent.agent import run_agent
from src.tools.data_loader import load_file
from src.database.db import init_db
from src.database.session_manager import SessionManager

# ── 页面配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMate — 智能数据分析助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS 样式 ──────────────────────────────────────────
st.markdown("""
<style>
    .data-status { padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; }
    .data-loaded { background: #d4edda; color: #155724; }
    .data-empty { background: #e2e3e5; color: #383d41; }
    .stat-card { background: #f8f9fa; padding: 0.6rem 0.8rem; border-radius: 0.4rem;
                 border: 1px solid #dee2e6; margin-bottom: 0.4rem; }
    .stat-label { font-size: 0.75rem; color: #6c757d; }
    .stat-value { font-size: 1.05rem; font-weight: 600; }
    .session-item { padding: 0.4rem 0.6rem; border-radius: 0.3rem; margin: 0.2rem 0;
                    cursor: pointer; border: 1px solid #dee2e6; }
    .session-item:hover { background: #e9ecef; }
    .session-active { background: #cfe2ff; border-color: #9ec5fe; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def save_current_session():
    """
    将当前会话状态持久化到 PostgreSQL

    流程:
      1. 更新会话元数据（df_name, temperature）
      2. 将 session_state.messages 中未持久化的消息逐条 INSERT 入库
      3. 所有消息标记为 _persisted = True，避免重复写入
    """
    if not st.session_state.session_id:
        return
    # 同步元数据
    SessionManager.update_meta(
        st.session_state.session_id,
        df_name=st.session_state.df_name,
        temperature=st.session_state.temperature,
    )
    # 只持久化新增的消息（通过 _persisted 标记判断）
    for msg in st.session_state.messages:
        if not msg.get("_persisted"):
            SessionManager.add_message(
                st.session_state.session_id, msg["role"], msg["content"]
            )
    # 标记全部消息为已持久化
    for m in st.session_state.messages:
        m["_persisted"] = True


def load_session(session_id: str):
    """
    从 PostgreSQL 加载指定会话的完整数据到 session_state

    Args:
        session_id: 目标会话 UUID

    加载内容:
      - messages（全部历史消息）
      - df_name / temperature / session_id
      - df 重置为 None（需用户重新上传数据文件）
    """
    data = SessionManager.get_session(session_id)
    if not data:
        return
    st.session_state.session_id = data["id"]
    st.session_state.messages = data["messages"]
    # 标记所有已加载消息为已持久化，避免重复写入
    for m in st.session_state.messages:
        m["_persisted"] = True
    st.session_state.df_name = data["df_name"]
    st.session_state.temperature = data["temperature"]
    # DataFrame 不入库，需重新上传
    st.session_state.df = None
    ToolContext.clear()


def launch_new_session():
    """
    创建全新的空白会话

    在 PostgreSQL 中创建记录，并清空当前 session_state 中所有分析状态。
    """
    sid = SessionManager.create_session()
    st.session_state.session_id = sid
    st.session_state.messages = []
    st.session_state.df = None
    st.session_state.df_name = None
    st.session_state.temperature = 0.1
    ToolContext.clear()


def refresh_session_list():
    """刷新侧边栏的会话列表缓存"""
    st.session_state.sessions = SessionManager.list_sessions()


# ══════════════════════════════════════════════════════════════
# 应用初始化
# ══════════════════════════════════════════════════════════════

# 启动时自动建表
init_db()

# 首次访问：恢复最近的会话，或无会话时创建默认会话
if "session_id" not in st.session_state:
    sessions = SessionManager.list_sessions()
    if sessions:
        # 有历史会话 → 自动加载最近更新的
        load_session(sessions[0]["id"])
    else:
        # 首次使用 → 创建默认会话
        launch_new_session()

# 加载会话列表供侧边栏展示
if "sessions" not in st.session_state:
    refresh_session_list()


# ══════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    # ── 品牌标识 ──
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=56)
    st.markdown("## DataMate")
    st.caption("基于 DeepSeek 的智能数据分析助手")
    st.divider()

    # ── 会话管理区 ──
    st.subheader("📂 会话管理")

    # 重命名栏：输入框 + 确认按钮
    c1, c2 = st.columns([3, 1])
    with c1:
        session_name = st.text_input(
            "会话名称",
            value="",
            placeholder="输入会话名称...",
            label_visibility="collapsed",
            key="session_name_input",
        )
    with c2:
        if st.button("✏️", help="重命名", use_container_width=True):
            if session_name.strip():
                SessionManager.rename_session(st.session_state.session_id, session_name.strip())
                refresh_session_list()
                st.rerun()

    # 会话列表（可折叠展开）
    with st.expander("会话列表", expanded=True):
        sessions = st.session_state.sessions
        if sessions:
            # 逐个渲染会话按钮
            for s in sessions:
                is_active = s["id"] == st.session_state.session_id
                label = f"{'🔵 ' if is_active else '⚪ '}{s['name']}"
                caption = f"{s['updated_at']} · {s['message_count']}条消息"
                if st.button(label, key=f"session_{s['id']}", use_container_width=True, help=caption):
                    # 点击非当前会话时：存档 → 加载目标 → 刷新
                    if s["id"] != st.session_state.session_id:
                        save_current_session()
                        load_session(s["id"])
                        refresh_session_list()
                        st.rerun()

        # 操作按钮：新建 / 删除
        col_new, col_del = st.columns(2)
        with col_new:
            if st.button("➕ 新建会话", use_container_width=True):
                save_current_session()
                launch_new_session()
                refresh_session_list()
                st.rerun()
        with col_del:
            if st.button("🗑️ 删除当前", use_container_width=True):
                sid = st.session_state.session_id
                SessionManager.delete_session(sid)
                refresh_session_list()
                # 删除后自动切换到剩余的第一个会话（或无则新建）
                remaining = SessionManager.list_sessions()
                if remaining:
                    load_session(remaining[0]["id"])
                else:
                    launch_new_session()
                st.rerun()

    st.divider()

    # ── 数据加载区 ──
    st.subheader("📁 数据加载")
    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=["csv", "xls", "xlsx", "json"],
        help="支持 CSV / Excel / JSON 格式，最大 200MB",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("🔄 加载数据", use_container_width=True):
            with st.spinner("正在解析数据..."):
                df = load_file(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.df_name = uploaded_file.name
                    # 同步工具上下文，让 Agent 工具可以访问数据
                    ToolContext.set(df, uploaded_file.name)
                    # 同步数据文件名到数据库
                    SessionManager.update_meta(st.session_state.session_id, df_name=uploaded_file.name)
                    # 加载新数据时清空当前对话历史
                    st.session_state.messages = []
                    st.rerun()

    # ── 数据概览区（仅在加载数据后显示）──
    if st.session_state.df is not None:
        st.divider()
        st.subheader("📋 数据概览")
        df = st.session_state.df

        # 文件名 + 规模卡片
        cols = st.columns(2)
        cols[0].markdown(
            f'<div class="stat-card"><span class="stat-label">文件名</span><br>'
            f'<span class="stat-value">{st.session_state.df_name}</span></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f'<div class="stat-card"><span class="stat-label">数据规模</span><br>'
            f'<span class="stat-value">{df.shape[0]:,} 行 x {df.shape[1]} 列</span></div>',
            unsafe_allow_html=True,
        )

        # 缺失值警告
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            st.warning(f"⚠️ {missing_total} 个缺失值")

        # 列名与类型表
        with st.expander("🔍 查看列名与类型"):
            dtype_df = pd.DataFrame({
                "列名": df.columns,
                "类型": df.dtypes.astype(str).values,
                "缺失": df.isnull().sum().values,
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        # 前100行预览
        with st.expander("🧾 前 100 行预览"):
            st.dataframe(df.head(100), use_container_width=True)

        # 移除数据按钮
        st.divider()
        if st.button("🗑️ 移除数据", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_name = None
            st.session_state.messages = []
            ToolContext.clear()
            SessionManager.update_meta(st.session_state.session_id, df_name=None)
            st.rerun()

    else:
        st.divider()
        st.info("💡 请先上传数据文件（CSV / Excel / JSON）")

    # ── 设置区 ──
    st.divider()
    st.subheader("⚙️ 设置")

    # Temperature 滑块
    col_t, col_b = st.columns([3, 1])
    with col_t:
        temp = st.slider(
            "创造性 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
        )
        st.session_state.temperature = temp
        # 实时同步到数据库
        SessionManager.update_meta(st.session_state.session_id, temperature=temp)
    with col_b:
        st.caption(f"**{temp:.2f}**")

    st.divider()
    st.caption(f"模型: `{DEEPSEEK_MODEL}`")

    # 清空对话（不删会话，只清内存中的消息）
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
# 主区域
# ══════════════════════════════════════════════════════════════

# ── 标题栏 + 数据状态 ──
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("📊 DataMate — 智能数据分析助手")
with col_status:
    if st.session_state.df is not None:
        st.markdown(
            '<div class="data-status data-loaded">✅ 数据已加载</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="data-status data-empty">⏳ 等待数据</div>',
            unsafe_allow_html=True,
        )

# 显示当前会话 ID（截取前8位）
current_sid = st.session_state.session_id
st.caption(f"会话: `{current_sid[:8]}...` | 用自然语言分析你的数据")

# ── 引导提示（无数据时）──
if st.session_state.df is None:
    st.info(
        "👈 **从左侧边栏上传一个数据文件开始分析**\n\n"
        "支持 CSV、Excel (xlsx/xls)、JSON 格式。"
        "上传后点击「加载数据」按钮即可开始对话。"
    )

# ── 聊天消息渲染 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 聊天输入区 ──
if st.session_state.df is not None:
    placeholder = "例如：分析各列的相关性 / 画一下月度趋势图 / 查找缺失值最多的列..."
else:
    placeholder = "请先在左侧上传数据文件..."

if prompt := st.chat_input(placeholder=placeholder):
    # 1. 用户消息持久化 + 界面追加
    SessionManager.add_message(st.session_state.session_id, "user", prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "_persisted": True}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 构建历史上下文（排除当前消息）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # 3. 调用 ReAct Agent 执行分析
    with st.chat_message("assistant"):
        with st.spinner("Agent 分析中，正在调用工具..."):
            try:
                result = run_agent(
                    prompt=prompt,
                    history=history,
                    temperature=st.session_state.temperature,
                )

                # 4. 渲染 Markdown 文本报告
                st.markdown(result["text"])

                # 5. 渲染可视化图表
                for img_path in result["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)

                # 6. 构建完整的持久化内容（文本 + 图表引用）
                full_content = result["text"]
                if result["images"]:
                    full_content += "\n\n---\n### 生成的图表\n"
                    for img_path in result["images"]:
                        full_content += f"\n![图表]({img_path})"

                # 7. 持久化 AI 回复
                SessionManager.add_message(
                    st.session_state.session_id, "assistant", full_content
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_content, "_persisted": True}
                )
            except Exception as e:
                error_msg = f"❌ Agent 执行出错: {str(e)}"
                st.error(error_msg)
                SessionManager.add_message(
                    st.session_state.session_id, "assistant", error_msg
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "_persisted": True}
                )
