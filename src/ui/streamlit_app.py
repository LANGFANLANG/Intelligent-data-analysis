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
from datetime import datetime, timezone

from src.config import DEEPSEEK_MODEL
from src.agent.context import ToolContext
from src.agent.agent import run_agent, run_agent_stream
from src.agent.router import route_stream
from src.tools.data_loader import load_file
from src.database.session_manager import SessionManager
from src.observability import ObservationContext, get_langfuse, score_trace
from src.agent.sanitizer import sanitize
from src.connectors import create_connector, get_connector, close_connector
from src.schema import get_cached_schema, refresh_schema
from src.logger import get_logger

_log = get_logger("ui")

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

# 首次访问：恢复最近的会话，或无会话时创建默认会话
if "session_id" not in st.session_state:
    sessions = SessionManager.list_sessions()
    if sessions:
        load_session(sessions[0]["id"])
    else:
        launch_new_session()

# 数据库连接状态
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "db_schema" not in st.session_state:
    st.session_state.db_schema = {}
if "db_tables" not in st.session_state:
    st.session_state.db_tables = []
if "db_error" not in st.session_state:
    st.session_state.db_error = ""

# 自动连接数据库（仅首次访问时执行）
if not st.session_state.db_connected:
    connector = create_connector()
    if connector and connector.is_connected:
        st.session_state.db_connected = True
        st.session_state.db_error = ""
        schema = get_cached_schema()
        st.session_state.db_schema = schema
        st.session_state.db_tables = list(schema.keys())
    else:
        st.session_state.db_connected = False

# 加载会话列表供侧边栏展示
if "sessions" not in st.session_state:
    refresh_session_list()

# 动态计算活跃数据源: 文件优先，数据库次之
has_file_data = st.session_state.df is not None
if has_file_data:
    ToolContext.set_data_source("file")
else:
    ToolContext.set_data_source("database")


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

    # ── 文件上传（始终可见）──
    st.subheader("📁 上传数据文件")
    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=["csv", "xls", "xlsx", "json"],
        help="支持 CSV / Excel / JSON 格式",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("🔄 加载数据", use_container_width=True):
            with st.spinner("正在解析数据..."):
                df = load_file(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.df_name = uploaded_file.name
                    ToolContext.set(df, uploaded_file.name)
                    ToolContext.set_data_source("file")
                    SessionManager.update_meta(st.session_state.session_id, df_name=uploaded_file.name)
                    st.session_state.messages = []
                    st.rerun()

    if st.session_state.df is not None:
        st.divider()
        st.subheader("📋 数据概览")
        df = st.session_state.df

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

        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            st.warning(f"⚠️ {missing_total} 个缺失值")

        with st.expander("🔍 查看列名与类型"):
            dtype_df = pd.DataFrame({
                "列名": df.columns,
                "类型": df.dtypes.astype(str).values,
                "缺失": df.isnull().sum().values,
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        with st.expander("🧾 前 100 行预览"):
            st.dataframe(df.head(100), use_container_width=True)

        st.divider()
        if st.button("🗑️ 移除数据", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_name = None
            st.session_state.messages = []
            ToolContext.clear()
            SessionManager.update_meta(st.session_state.session_id, df_name=None)
            st.rerun()

    # ── 数据库连接（自动连接，始终可见）──
    if st.session_state.db_connected:
        from src.config import DB_HOST, DB_PORT, DB_NAME
        from src.config import DB_USER

        st.divider()
        st.subheader("🗄️ 数据库")
        st.success(f"已连接: {DB_NAME} ({DB_HOST}:{DB_PORT})")

        table_count = len(st.session_state.db_tables)
        cols = st.columns(2)
        cols[0].markdown(
            f'<div class="stat-card"><span class="stat-label">数据库</span><br>'
            f'<span class="stat-value">{DB_NAME}</span></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f'<div class="stat-card"><span class="stat-label">表数量</span><br>'
            f'<span class="stat-value">{table_count} 张表</span></div>',
            unsafe_allow_html=True,
        )

        with st.expander("📊 数据表浏览", expanded=False):
            search_term = st.text_input(
                "搜索表名",
                placeholder="输入关键词筛选...",
                label_visibility="collapsed",
                key="table_search",
            )

            tables = st.session_state.db_tables
            if search_term:
                tables = [t for t in tables if search_term.lower() in t.lower()]

            if tables:
                schema = st.session_state.db_schema
                for table_name in tables[:20]:
                    info = schema.get(table_name, {})
                    cols_list = info.get("columns", [])
                    row_count = info.get("row_count")

                    with st.expander(f"📄 {table_name}" + (f" (~{row_count:,} 行)" if row_count else "")):
                        if cols_list:
                            preview_data = []
                            for c in cols_list:
                                pk = "🔑" if c.get("is_primary_key") else ""
                                preview_data.append({
                                    "列名": c["name"],
                                    "类型": c["type"],
                                    "可空": "YES" if c.get("nullable") else "NO",
                                    "键": pk,
                                })
                            st.dataframe(
                                pd.DataFrame(preview_data),
                                use_container_width=True,
                                hide_index=True,
                            )

                        if st.button("📋 预览数据", key=f"preview_{table_name}"):
                            try:
                                connector = get_connector()
                                if connector:
                                    df = connector.execute_query(
                                        f"SELECT * FROM `{table_name}` LIMIT 20"
                                    )
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"预览失败: {e}")

                if len(st.session_state.db_tables) > 20:
                    st.caption(f"还有 {len(st.session_state.db_tables) - 20} 张表未显示，使用搜索筛选")
            else:
                st.info("未找到匹配的表")

            if st.button("🔄 刷新", use_container_width=True, key="refresh_schema_btn"):
                with st.spinner("正在刷新..."):
                    schema = refresh_schema()
                    st.session_state.db_schema = schema
                    st.session_state.db_tables = list(schema.keys())
                st.rerun()

    # ── 活跃数据源指示 ──
    st.divider()
    if has_file_data:
        st.caption("📊 当前活跃数据源: 📁 上传文件")
    elif st.session_state.db_connected:
        st.caption("📊 当前活跃数据源: 🗄️ 数据库")
    else:
        st.caption("📊 当前活跃数据源: ⏳ 无")

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
    if has_file_data:
        st.markdown(
            '<div class="data-status data-loaded">✅ 数据已加载</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.db_connected:
        st.markdown(
            '<div class="data-status data-loaded">✅ 数据库已连接</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="data-status data-empty">⏳ 等待数据</div>',
            unsafe_allow_html=True,
        )

current_sid = st.session_state.session_id
st.caption(f"会话: `{current_sid[:8]}...` | 用自然语言分析你的数据")

has_active_data = has_file_data or st.session_state.db_connected

if not has_active_data:
    st.info(
        "👈 **从左侧边栏上传数据文件开始分析**\n\n"
        "支持 CSV、Excel (xlsx/xls)、JSON 格式。"
        "上传后点击「加载数据」按钮即可开始对话。"
    )

# ── 聊天消息渲染 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 渲染消息附带的图表（支持历史会话恢复后显示）
        for img_path in msg.get("images", []):
            if os.path.exists(img_path) and os.path.getsize(img_path) >= 1024:
                st.image(img_path, use_container_width=True)
            else:
                st.caption(f"[图表不可用: {os.path.basename(img_path)}]")

# ── 聊天输入区 ──
if has_active_data:
    placeholder = "例如：分析各列的相关性 / 画一下月度趋势图 / 查找缺失值最多的列..."
else:
    placeholder = "请先在左侧上传数据文件或连接数据库..."

if prompt := st.chat_input(placeholder=placeholder):
    # Prompt 注入防护
    try:
        prompt = sanitize(prompt)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # 初始化 Langfuse 观测上下文
    ObservationContext.init(
        session_id=st.session_state.session_id,
        user_id="default",
    )

    # 0. 首次提问时用问题内容自动命名会话，时间标记为首次交互时间
    if len(st.session_state.messages) == 0:
        new_name = prompt[:40] + ("..." if len(prompt) > 40 else "")
        SessionManager.rename_session(st.session_state.session_id, new_name)
        SessionManager.update_meta(
            st.session_state.session_id,
            created_at=datetime.now(timezone.utc),
        )
        refresh_session_list()

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

    # 3. 流式调用 ReAct Agent 执行分析
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        status_placeholder = st.empty()

        text_buffer = ""
        images = []
        full_text = ""
        had_error = False
        is_db = not has_file_data and st.session_state.db_connected
        db_status_shown = False

        try:
            for event in route_stream(
                prompt=prompt,
                history=history,
                temperature=st.session_state.temperature,
            ):
                if event["type"] == "token":
                    text_buffer += event["content"]
                    text_placeholder.markdown(text_buffer)

                elif event["type"] == "tool_start":
                    if is_db:
                        if not db_status_shown:
                            status_placeholder.info("正在分析数据库...")
                            db_status_shown = True
                    else:
                        status_placeholder.info(f"正在调用工具: {event['tool']}...")

                elif event["type"] == "tool_end":
                    if not is_db:
                        status_placeholder.empty()
                    for img_path in event.get("images", []):
                        if os.path.exists(img_path) and os.path.getsize(img_path) >= 1024:
                            st.image(img_path, use_container_width=True)
                        else:
                            st.caption(f"[图表不可用: {os.path.basename(img_path)}]")
                    images.extend(event.get("images", []))

                elif event["type"] == "error":
                    had_error = True
                    text_placeholder.error(event["content"])
                    full_text = event["content"]
                    break

                elif event["type"] == "done":
                    full_text = event.get("full_text", text_buffer)
                    images = event.get("images", images)

            status_placeholder.empty()
            # 确保最终文本一致
            if full_text and text_buffer and not full_text.startswith(text_buffer):
                text_placeholder.markdown(full_text)

        except Exception as e:
            had_error = True
            error_str = str(e)
            if "recursion" in error_str.lower() or "limit" in error_str.lower():
                error_msg = "分析步骤过多，建议将问题拆分为多个小问题依次提问"
            elif "timeout" in error_str.lower():
                error_msg = "分析超时，请简化问题或检查数据质量"
            elif "api" in error_str.lower() or "401" in error_str.lower() or "403" in error_str.lower():
                error_msg = "模型连接失败，请检查 API Key 或网络连接"
            elif "rate" in error_str.lower() or "429" in error_str.lower():
                error_msg = "API 请求过于频繁，请稍后再试"
            else:
                error_msg = f"分析失败: {error_str}"
            text_placeholder.error(error_msg)
            full_text = error_msg

        # 4. 持久化 AI 回复
        if full_text:
            SessionManager.add_message(
                st.session_state.session_id, "assistant", full_text, images
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "images": images,
                "_persisted": True,
            })

        # 5. Langfuse 观测：flush 上报 + 显示追踪链接 + 用户反馈
        langfuse_client = get_langfuse()
        if langfuse_client is not None:
            try:
                langfuse_client.flush()
            except Exception:
                pass
            trace_id = ObservationContext.get_trace_id()
            trace_url = f"http://localhost:3000/trace/{trace_id}" if trace_id else None

            col_link, col_fb = st.columns([4, 1])
            with col_link:
                if trace_url:
                    st.caption(f"[🔍 在 Langfuse 中查看完整调用链]({trace_url})")
                elif trace_id:
                    st.caption(f"Trace: `{trace_id[:12]}...`")
            with col_fb:
                fb_key = f"fb_{len(st.session_state.messages)}"
                if st.button("👍", key=f"up_{fb_key}", help="回答有帮助"):
                    if trace_id:
                        score_trace(trace_id, name="user_feedback", value=1.0)
                    st.toast("感谢反馈！", icon="✅")
                if st.button("👎", key=f"down_{fb_key}", help="回答需改进"):
                    if trace_id:
                        score_trace(trace_id, name="user_feedback", value=0.0)
                    st.toast("感谢反馈，我们会持续改进！", icon="📝")

        # 刷新会话列表（名称可能已更新）
        refresh_session_list()
