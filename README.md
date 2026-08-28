# 漫译 · 漫画多语言智能翻译系统

一键翻译漫画气泡文字，保持原排版不改变。支持 **日语/英语 → 中文**，专有名词自定义词典。

## 功能

- 🖼️ 上传漫画图片（JPG / PNG / WebP / BMP，≤10MB）
- 🔍 真实 OCR 文字识别（PaddleOCR，支持日/英/中）
- 🔄 翻译流水线：文本检测 → OCR → 翻译 → 图像修复 → 渲染
- 📐 保持原气泡位置与尺寸，自动排版中文译文
- 📊 分步进度条实时反馈（检测/识别/翻译/修复/渲染）
- 📥 网页预览 + 下载翻译结果
- 📖 专有名词管理：内置词库 + 图形化录入 + JSON 批量导入

## 快速开始

### 方式一：手动启动

```bash
# 1. 后端（基础依赖）
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r backend/requirements.txt

# 2. 真实 OCR（可选但推荐，启用真实文字识别）
pip install -r backend/requirements-ai.txt

# 3. 前端
cd frontend
npm install
npm run build
cd ..

# 4. 启动
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
| `MANGA_TRANSLATOR_BACKEND` | 翻译后端：`google` / `mymemory` / `deepl` / `openai` | `google` |
| `MANGA_DEEPL_AUTH_KEY` | DeepL API Key（可选） | 空 |
| `MANGA_OPENAI_API_KEY` | OpenAI 兼容接口 Key（可选） | 空 |
| `MANGA_MAX_UPLOAD_MB` | 单张图片大小上限 | `10` |

> **DeepL**：需同时设置 `MANGA_TRANSLATOR_BACKEND=deepl` 与 `MANGA_DEEPL_AUTH_KEY`（免费版 Key 以 `:fx` 结尾，走 `api-free.deepl.com`）。只设后端不配 Key 时，翻译会自动降级到 Google。
>
> 翻译器内置自动降级链：配置后端 → Google → MyMemory → 仅词典替换，保证网络不佳时也能运行。

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
        translator.py   # 翻译（Google/MyMemory/DeepL/OpenAI）
        inpainter.py    # 图像修复
        renderer.py     # 渲染排版
    storage/       # SQLite + 文件存储
docs/             # PRD / 竞品分析 / 功能矩阵
```

## 翻译流水线

```
上传图片
  → 文本检测   定位气泡文字区域（PaddleOCR 检测或 OpenCV）
  → OCR        识别文字内容（PaddleOCR 真实识别）
  → 翻译       按语言对翻译（自动应用专有名词词典）
  → 图像修复   擦除原文字（OpenCV inpaint）
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
- **Demo 模式**：未安装 AI 依赖时，OCR 使用模拟识别，适合快速体验流程。
- 翻译默认使用 Google 免费接口（`dict-chrome-ex` 客户端），生产环境建议配置 DeepL 或 OpenAI Key 以获得更高稳定性。
