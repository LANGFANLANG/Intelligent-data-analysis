# DataMate 项目全局规则

> 本文件包含项目开发过程中必须遵守的约定和规范。
> AI 助手在处理本项目时，应在每次对话中遵循以下规则。


## 1. 代码规范

### 必须写注释
- 所有 `.py` 文件必须包含完整的**中文注释**
- 每个模块开头应有模块说明（用途、包含的类/函数）
- 每个类应有类说明（职责、使用场景）
- 每个函数/方法应有文档字符串（Args、Returns、功能描述）
- 关键代码块应有行内注释（说明逻辑意图，而非逐行翻译代码）


## 2. 语言规范

### 回答用中文
- 所有对用户的回复必须使用**中文**
- 代码中的字符串、提示词、UI 文案全部使用中文
- 异常消息、日志输出使用中文
- 变量名、函数名、类名使用英文（Python 社区惯例）


## 3. 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| LLM | DeepSeek V4 Flash | 通过 LangChain ChatOpenAI 兼容接入 |
| 编排 | LangChain | Agent 循环 + Tool 注册 |
| 数据库 | PostgreSQL 16 | Docker 部署，SQLAlchemy ORM |
| 前端 | Streamlit | 对话式 UI |
| 包管理 | uv | 依赖管理 + 虚拟环境 |
| 容器 | Docker Compose | 本地 PostgreSQL 服务 |


## 4. 目录结构

```
data-analysis-agent/
├── src/
│   ├── config.py              # 全局配置
│   ├── llm/                   # LLM 客户端
│   │   └── deepseek_client.py
│   ├── agent/                  # Agent 核心（ReAct 编排）
│   │   ├── context.py           # 工具上下文（DataFrame 单例）
│   │   ├── tools.py             # LangChain Tool 注册表
│   │   └── agent.py             # ReAct Agent 组装 + run_agent()
│   ├── database/              # 数据库层（会话管理 + ORM）
│   │   ├── models.py          # ORM 模型
│   │   ├── db.py              # 数据库连接
│   │   └── session_manager.py # 会话 CRUD
│   ├── tools/                 # 工具层
│   │   ├── data_loader.py     # CSV/Excel/JSON 加载
│   │   ├── data_cleaner.py    # 缺失值/去重/类型转换
│   │   ├── analyzer.py        # describe/corr/groupby
│   │   └── visualizer.py      # 6种图表生成
│   └── ui/                    # 前端界面
│       └── streamlit_app.py
├── docker-compose.yml         # PostgreSQL 容器
├── pyproject.toml             # uv 项目配置
├── run.py                     # 入口脚本
├── .env                       # 环境变量
└── RULES.md                   # 本文件
```


## 5. 会话数据模型

```
sessions (会话表)
├── id UUID PK
├── name VARCHAR(255)
├── df_name VARCHAR(500)
├── temperature FLOAT
├── created_at TIMESTAMPTZ
└── updated_at TIMESTAMPTZ

messages (消息表)
├── id BIGSERIAL PK
├── session_id UUID FK → sessions(id) ON DELETE CASCADE
├── role VARCHAR(20) CHECK (user|assistant)
├── content TEXT
└── created_at TIMESTAMPTZ
```


## 6. 启动命令

```bash
# 启动数据库
docker compose up -d

# 启动应用
uv run python run.py          # Streamlit 界面
uv run python run.py --cli    # 命令行模式
```
