# DataMate — 智能数据分析助手

基于 **DeepSeek** + **LangChain/LangGraph** + **Streamlit** 的智能数据分析对话助手。

上传 CSV/Excel/JSON 数据，用自然语言提问，Agent 自动调用 18 种分析工具完成统计、清洗、可视化，并逐字流式输出结果。

---

## 功能特性

- **自然语言驱动** — 中文提问、Agent 自主选择工具，零代码完成数据分析
- **17 种分析工具** — 描述性统计、相关性、分组聚合、缺失值处理、去重等
- **6 种可视化图表** — 折线图、柱状图、散点图、饼图、热力图、直方图，支持中文字体
- **Token 级流式输出** — 逐字蹦出分析结果，工具调用实时可见
- **意图识别路由** — 自动区分闲聊和分析请求，无数据时走纯聊天路径
- **全链路追踪** — 记录每次请求的完整调用树（分类→LLM 调用→工具执行），存入 PostgreSQL
- **会话持久化** — 对话历史保存到数据库，支持多会话切换
- **大数据保护** — 超过 50k 行自动采样计算，防止 OOM
- **安全防护** — 限流器、API 重试、Prompt 注入检测

---

## 快速开始

### 1. 准备环境

```bash
# 克隆项目
git clone <repo-url>
cd data-analysis-agent

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 2. 建表（手动）

```bash
# 连接 PostgreSQL 执行 SQL
psql -h localhost -U data_agent -d data_agent -f init.sql
```

### 3. 一键启动

```bash
docker compose up -d
# → Streamlit: http://localhost:8501
```

### 4. 本地开发

```bash
# 安装依赖
uv sync

# 启动 PostgreSQL
docker compose up -d postgres

# 启动应用
uv run python run.py
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek (Chat API) |
| Agent 框架 | LangGraph (ReAct Agent) |
| UI | Streamlit |
| 数据库 | PostgreSQL 16 |
| ORM | SQLAlchemy 2.0 |
| 可视化 | Matplotlib + Seaborn |
| 数据 | Pandas + NumPy + openpyxl |
| 日志 | Python logging (JSON 格式) |
| 部署 | Docker + docker-compose |

## 架构

```
用户输入
  ↓
sanitizer (注入防护)
  ↓
classify_intent (意图分类)
  ├── "chat"      → chat_llm_stream (纯对话)
  └── "analysis"  → ReAct Agent (17 tools)
                        ├── LLM 调用 (DeepSeek)
                        ├── 工具调用 (data_cleaner / analyzer / visualizer)
                        └── 图表生成 (Matplotlib → outputs/*.png)
  ↓
streamlit_app (逐 token 渐进渲染 + 图片即刻显示)
  ↓
TraceManager (全链路 span 写入 PostgreSQL)
```

## 目录结构

```
data-analysis-agent/
├── run.py                  # 入口 (Streamlit / CLI)
├── Dockerfile              # 应用容器
├── docker-compose.yml      # app + postgres 双服务
├── pyproject.toml          # uv 项目配置
├── .env.example            # 环境变量模板
├── src/
│   ├── agent/              # Agent 核心
│   │   ├── agent.py        #   ReAct Agent + Chat Agent
│   │   ├── router.py       #   意图识别 + 路由分发
│   │   ├── sanitizer.py    #   Prompt 注入防护
│   │   ├── context.py      #   工具上下文 (contextvars 隔离)
│   │   └── tools.py        #   17 个 LangChain Tool 注册
│   ├── llm/                # LLM 客户端
│   │   ├── deepseek_client.py  # DeepSeek API 封装
│   │   └── rate_limiter.py     # 令牌桶限流
│   ├── tools/              # 工具实现
│   │   ├── data_loader.py  #   文件加载 + 校验
│   │   ├── data_cleaner.py #   数据清洗
│   │   ├── analyzer.py     #   统计分析
│   │   └── visualizer.py   #   Matplotlib 图表
│   ├── ui/
│   │   └── streamlit_app.py # Streamlit 前端
│   ├── database/           # 持久化
│   │   ├── models.py       #   ORM 模型 (Session/Message/Trace)
│   │   ├── db.py           #   连接池管理
│   │   └── session_manager.py # 会话 CRUD
│   ├── tracing/            # 全链路追踪
│   │   └── trace.py
│   ├── config.py           # 全局配置
│   └── logger.py           # 结构化日志
├── outputs/                # 生成的图表 PNG
├── logs/                   # 应用日志 (JSON)
└── RULES.md                # 开发者规范
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key (必填) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API 地址 |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL 连接串 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `AGENT_MAX_STEPS` | `40` | Agent 最大步骤数 |
| `AGENT_TIMEOUT_SECONDS` | `120` | 执行超时 |
| `LLM_RATE_LIMIT` | `30` | 每分钟 API 请求上限 |
| `SAMPLE_MAX_ROWS` | `50000` | 自动采样阈值 |
| `MAX_FILE_SIZE_MB` | `100` | 上传文件大小上限 |
| `STREAMLIT_PORT` | `8501` | 服务端口 |
