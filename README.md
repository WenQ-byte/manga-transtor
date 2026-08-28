# 漫译 · 漫画多语言智能翻译系统

一键翻译漫画气泡文字，保持原排版不改变。支持 **日语/英语 → 中文**，专有名词自定义词典。

## 功能

- 上传漫画图片（JPG / PNG / WebP / BMP，≤10MB）
- 真实 OCR 文字识别（PaddleOCR，支持日/英/中）
- 翻译流水线：文本检测 → OCR → 翻译 → 图像修复 → 渲染
- 保持原气泡位置与尺寸，自动排版中文译文
- 分步进度条实时反馈（检测/识别/翻译/修复/渲染）
- 网页预览 + 下载翻译结果
- 专有名词管理：内置词库 + 图形化录入 + JSON 批量导入

## 快速开始

### 方式一：手动启动

```bash
# 1. 后端（基础依赖）
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r backend/requirements.txt

# 2. 真实 OCR（可选但推荐）
pip install -r backend/requirements-ai.txt

# 3. LaMa 神经修复（可选，效果更好）
pip install -r backend/requirements-inpaint.txt

# 4. 前端
cd frontend
npm install
npm run build
cd ..

# 5. 启动
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

### 方式二：Docker

```bash
docker compose up -d --build
```

访问 http://localhost:8000

## 配置

复制 `.env.example` 为 `.env` 并按需修改：

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `MANGA_PIPELINE_MODE` | 流水线模式：`real`（真实OCR）/ `demo`（无需模型） | `real` |
| `MANGA_TRANSLATOR_BACKEND` | 翻译后端：`deepseek` / `google` / `mymemory` / `deepl` / `openai` | `deepseek` |
| `MANGA_DEEPSEEK_API_KEY` | DeepSeek API Key（推荐，漫画口语翻译最自然） | 空 |
| `MANGA_DEEPSEEK_MODEL` | DeepSeek 模型 | `deepseek-chat` |
| `MANGA_DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `MANGA_DEEPL_AUTH_KEY` | DeepL API Key（可选，口语翻译较差） | 空 |
| `MANGA_OPENAI_API_KEY` | OpenAI 兼容接口 Key（可选） | 空 |
| `MANGA_INPAINTER_BACKEND` | 修复引擎：`lama`（神经网络）/ `cv`（无模型） | `cv` |
| `MANGA_LAMA_MODEL_PATH` | LaMa 权重路径（空则自动搜索） | 空 |
| `MANGA_LAMA_INPAINT_SIZE` | LaMa 推理缩放上限（像素） | `1024` |
| `MANGA_INPAINT_DEVICE` | 修复设备：`cpu` / `cuda` | `cpu` |
| `MANGA_MAX_UPLOAD_MB` | 单张图片大小上限 | `10` |

> **翻译后端推荐**：DeepSeek（`deepseek-chat`）对漫画口语/语气词理解力最佳，需设置 `MANGA_DEEPSEEK_API_KEY`。免费后端（Google / MyMemory）作为 fallback，质量一般。
>
> **图像修复**：`cv` 无依赖但效果有限（适合白底气泡）；`lama` 需安装 `torch`（CPU 版约 200MB），效果对齐 manga-image-translator。

## 架构

```
frontend/          # React + Vite + Tailwind 前端（中文 UI）
backend/
  app/
    api/           # REST API（翻译 / 词典 / 文件）
    services/      # 翻译流水线（可插拔引擎）
      engines/
        detector.py     # 文本检测（OpenCV）
        ocr.py          # OCR（PaddleOCR 真实识别 / Demo）
        translator.py   # 翻译（DeepSeek/Google/MyMemory/DeepL/OpenAI）
        mask.py         # 精确笔画掩膜（poly + Otsu 自适应阈值）
        inpainter.py    # 图像修复（CV 无模型）
        lama.py         # LaMa 神经修复引擎
        lama_model.py   # LaMa 模型结构（FFC 生成网络）
        renderer.py     # 渲染排版
    storage/       # SQLite + 文件存储
docs/             # PRD / 竞品分析 / 功能矩阵
```

## 翻译流水线

```
上传图片
  → 文本检测   定位气泡文字区域（PaddleOCR 检测）
  → OCR        识别文字内容，保留完整检测多边形
  → 翻译       DeepSeek 口语化翻译（自动应用专有名词词典）
  → 图像修复   精确掩膜 + 神经修复擦除原文字
  → 渲染       将译文排版回原气泡位置
  → 下载结果
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/translate` | 上传图片创建翻译任务 |
| GET | `/api/translate/{id}/status` | 查询翻译进度 |
| GET | `/api/translate/{id}/result` | 下载翻译结果 |
| DELETE | `/api/translate/{id}` | 删除任务 |
| GET/POST | `/api/glossary` | 专有名词列表 / 新增 |
| PUT/DELETE | `/api/glossary/{id}` | 修改 / 删除词条 |
| POST | `/api/glossary/import` | JSON 批量导入 |
| GET | `/api/health` | 健康检查 |

在线 API 文档：`/docs`

## 说明

- **真实 OCR**：安装 `requirements-ai.txt` 后自动启用 PaddleOCR 真实文字识别（首次运行会自动下载模型）。
- **LaMa 修复**：安装 `requirements-inpaint.txt`（torch）后，设置 `MANGA_INPAINTER_BACKEND=lama` 启用神经修复，效果接近 manga-image-translator。
- **Demo 模式**：未安装 AI 依赖时，OCR 使用模拟识别，适合快速体验流程。
- 翻译默认使用 DeepSeek（`deepseek-chat`），免费后端（Google / MyMemory）作为自动 fallback。
