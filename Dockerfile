# 后端镜像
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（字体用于中文渲染）
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/app ./app

# 复制前端构建产物（如存在）
COPY frontend/dist ./frontend/dist

ENV MANGA_DATA_DIR=/data
ENV PYTHONPATH=/app

EXPOSE 8000

VOLUME ["/data"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
