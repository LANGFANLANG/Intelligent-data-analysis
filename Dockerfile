FROM python:3.11-slim

WORKDIR /app

# 系统依赖（matplotlib 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install uv --no-cache-dir

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 复制源码
COPY . .

# 创建运行时目录
RUN mkdir -p outputs logs

EXPOSE 8501

CMD ["uv", "run", "python", "run.py"]
