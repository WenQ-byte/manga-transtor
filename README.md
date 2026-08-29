# 漫译 · 漫画多语言智能翻译系统

一键翻译漫画气泡文字，保持原排版不改变。支持 **日语/英语 → 中文**，专有名词自定义词典。

## 功能

- 上传漫画图片（JPG / PNG / WebP / BMP，≤10MB）
- 真实 OCR 文字识别，**默认 manga-image-translator 检测/OCR 引擎**（漫画专训模型，复杂漫画/艺术字/竖排/网点底识别更准；GPL-3.0，见 `THIRD_PARTY.md`），PaddleOCR 为可切回退（`MANGA_OCR_BACKEND=paddle`）
- 默认 **ctd 检测器**（ComicTextDetector，复杂页召回强；可切 DBNet `MANGA_MIT_DETECTOR=default`）+ **mit48+manga-ocr 混合识别**（最准），置信度 ≥0.5 过滤噪声
- 翻译流水线：文本检测 → OCR → **图像修复** → 翻译 → 渲染
- **按气泡整块翻译**：同气泡对话多行合并成一次翻译请求，译文更地道；分组在擦除后的干净图上进行，同一气泡必成一组
- **整块擦除**：按文本多边形整块填充掩膜（零残留），背景由 cv / LaMa 重建
- 可选 **manga-ocr 识别**（`MANGA_OCR_BACKEND=mangaocr`/`mit48+mangaocr`）与**小字放大**（`MANGA_MIT_OCR_UPSCALE`），增强风格化字体/小字召回
- 横排 / 竖排自适应排版：整块均衡分列/分行（列长、行长相近），气泡内居中，文字不越界不丢失（掩膜不可信时回退矩形裁剪）
- 分步进度条实时反馈（检测/识别/修复/翻译/渲染）
- 网页预览 + 下载翻译结果
- 专有名词管理：内置词库 + 图形化录入 + JSON 批量导入（整词匹配）
- 玻璃拟态前端质感（GlassSurface / SpecularButton / ParticleText 等组件）

## 快速开始

### 方式一：一键启动（推荐）

安装依赖后，双击 `start.bat`（或运行 `.venv\Scripts\python.exe start.py`），按交互菜单选择启动后端 / 前端 / 前后端：

```bash
# 安装依赖
pip install -r backend/requirements.txt          # 核心
pip install -r backend/requirements-ai.txt       # 真实 OCR（PaddleOCR）
pip install -r backend/requirements-inpaint.txt  # LaMa 神经修复（可选，效果更好）
cd frontend && npm install
```

随后打开 `start.bat` 选择 `[1]` 启动后端，访问 http://localhost:8000；选择 `[3]` 同时启动前后端（Vite 开发模式 http://localhost:5173）。

### 方式二：手动启动

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

> 注：`--app-dir backend` 必须保留（模块是 `app.main:app`）。

### 方式三：Docker

```bash
docker compose up -d --build
```

访问 http://localhost:8000

## 配置

复制 `.env.example` 为 `.env` 并按需修改：

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `MANGA_PIPELINE_MODE` | 流水线模式：`real`（真实OCR）/ `demo`（无需模型） | `real` |
| `MANGA_TRANSLATOR_BACKEND` | 翻译后端：`deepseek` / `google` / `deepl` / `openai` / `mymemory` | `deepseek` |
| `MANGA_DEEPSEEK_API_KEY` | DeepSeek API Key（推荐，漫画口语翻译最自然） | 空 |
| `MANGA_DEEPSEEK_MODEL` | DeepSeek 模型 | `deepseek-chat` |
| `MANGA_DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `MANGA_DEEPL_AUTH_KEY` | DeepL API Key（可选，支持批量请求） | 空 |
| `MANGA_OPENAI_API_KEY` | OpenAI 兼容接口 Key（可选） | 空 |
| `MANGA_INPAINTER_BACKEND` | 修复引擎：`lama`（神经网络）/ `cv`（无模型） | `cv` |
| `MANGA_LAMA_MODEL_PATH` | LaMa 权重路径（空则自动搜索） | 空 |
| `MANGA_LAMA_INPAINT_SIZE` | LaMa 推理缩放上限（像素） | `1024` |
| `MANGA_INPAINT_DEVICE` | 修复设备：`cpu` / `cuda` | `cpu` |
| `MANGA_MAX_UPLOAD_MB` | 单张图片大小上限 | `10` |

> **翻译后端推荐**：DeepSeek（`deepseek-chat`）对漫画口语/语气词理解力最佳，需设置 `MANGA_DEEPSEEK_API_KEY`。免费后端（Google / MyMemory）作为 fallback，质量一般。Google 需使用 `dict-chrome-ex` 接口避免限流；DeepL 支持批量请求，一次调用翻译全部文本。默认回退链：Google → MyMemory → 仅词典替换。
>
> **图像修复**：`cv` 无依赖但效果有限（适合白底气泡）；`lama` 需安装 `torch`（CPU 版约 200MB），效果对齐 manga-image-translator。

## 架构

```
frontend/          # React + Vite + Tailwind 前端（中文 UI，玻璃拟态组件）
backend/
  app/
    api/           # REST API（翻译 / 词典 / 文件）
    services/      # 翻译流水线（可插拔引擎）
      engines/
        base.py         # 引擎基类与工厂
        detector.py     # 文本检测（OpenCV）
        ocr.py          # OCR（PaddleOCR 真实识别 / Demo，竖排旋转检测）
        translator.py   # 翻译（DeepSeek/Google/MyMemory/DeepL/OpenAI，批量+回退链）
        mask.py         # 精确笔画掩膜（poly + Otsu 自适应阈值）
        inpainter.py    # 图像修复（CV 无模型）
        bubble.py       # 气泡过滤（气泡外涂鸦丢弃）
        lama.py         # LaMa 神经修复引擎
        lama_model.py   # LaMa 模型结构（FFC 生成网络）
        renderer.py     # 渲染排版（气泡感知，横排/竖排自适应）
    storage/       # SQLite + 文件存储
docs/             # PRD / 竞品分析 / 功能矩阵
```

> 引擎通过 `services/engines/factory.py` 的 `get_engine(type)` 统一获取（lru_cache 缓存）。OCR 引擎自带检测（`supports_detection=True`）时跳过独立 detector，直接用 OCR 检测+识别。

## 翻译流水线

```
上传图片
  → 文本检测   定位气泡文字区域（置信度 ≥0.5 过滤 + 气泡过滤）
  → OCR        识别文字内容，保留完整检测多边形
  → 图像修复   整块掩膜擦除原文字（cv / LaMa 重建背景）
  → 翻译       气泡整块翻译（DeepSeek 最自然 / 批量请求 / 词典整词匹配）
  → 渲染       均衡排版回原气泡（横排分行、竖排分列，均居中且等长）
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

## 测试

```bash
.venv\Scripts\python.exe backend\tests\test_app.py
```

stdlib `unittest`（非 pytest）。测试会设 `MANGA_DATA_DIR` 到临时目录；`test_pipeline_runs` 走真实 OCR 路径，已装 AI 依赖时会加载 PaddleOCR 模型、较慢。

## 说明

- **真实 OCR**：安装 `requirements-ai.txt` 后自动启用 PaddleOCR 真实文字识别（首次运行会自动下载模型）。检测框置信度 <0.5 或空文本会被过滤。
- **LaMa 修复**：安装 `requirements-inpaint.txt`（torch）后，设置 `MANGA_INPAINTER_BACKEND=lama` 启用神经修复，效果接近 manga-image-translator。
- **Demo 模式**：未安装 AI 依赖时，OCR 使用模拟识别，适合快速体验流程。
- **翻译**：默认使用 DeepSeek（`deepseek-chat`），免费后端（Google / MyMemory）作为自动 fallback。Google 需用 `dict-chrome-ex` 接口防限流；DeepL 支持批量请求。
- **竖排渲染**：漫画竖排文字靠把原图旋转 90° 再检测、坐标映射回原图实现；渲染时逐列从右到左、整组居中，列间距与字距自适应防文字重叠。
