# PerpBot Dockerfile
# 多阶段构建，优化镜像大小

# ==================== 构建阶段 ====================
FROM python:3.11-slim as builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== 运行阶段 ====================
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# 创建非 root 用户
RUN groupadd --gid 1000 perpbot && \
    useradd --uid 1000 --gid perpbot --shell /bin/bash --create-home perpbot

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 复制应用代码
COPY --chown=perpbot:perpbot . .

# 创建数据目录
RUN mkdir -p /app/data/logs /app/data/tracking && \
    chown -R perpbot:perpbot /app/data

# 切换到非 root 用户
USER perpbot

# 健康检查
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/data/logs') else 1)"

# 默认命令
CMD ["python", "main.py"]