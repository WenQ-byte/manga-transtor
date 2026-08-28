# AGENTS.md

漫画多语言智能翻译系统（日语/英语 → 中文）。FastAPI 后端 + React/Vite 前端（Tailwind v4 + Framer Motion + Lucide） + PaddleOCR。

## 运行 / 测试命令

- 手动启动后端（必须加 `--app-dir backend`，模块是 `app.main:app`）：
  ```
  .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
  ```
- 前端：`cd frontend; npm run build`。**Windows 上 npm 用 `npm.cmd`，`npm.ps1` 被执行策略禁止。**
- 测试（stdlib `unittest`，非 pytest）：
  ```
  .venv\Scripts\python.exe backend\tests\test_app.py
  ```
  测试会设 `MANGA_DATA_DIR` 到临时目录；`test_pipeline_runs` 走真实 OCR 路径，若已装 AI 依赖会加载 PaddleOCR 模型、较慢。
- 无 linter / typecheck / formatter 配置，不需要跑。

## 依赖两档

- `backend/requirements.txt`：核心（FastAPI/Pillow/opencv/httpx）。仅装这个时 OCR 走 Demo 模式（模拟识别）。
- `backend/requirements-ai.txt`：`paddlepaddle==3.3.1` + `paddleocr==3.7.0`，装了才启用真实 OCR。
- 首次真实 OCR 会自动下载模型到 `~/.paddlex/official_models/`；可用 `PADDLEOCR_HOME` 指向 `backend/data/models`。

## 配置

- pydantic-settings，环境变量前缀 `MANGA_`，读 `.env`（见 `backend/app/config.py`）。如 `MANGA_PIPELINE_MODE`（real/demo）、`MANGA_TRANSLATOR_BACKEND`、`MANGA_MAX_UPLOAD_MB`。

## 架构要点（非文件名能看出的）

- 应用代码在 `backend/app`（不是 `backend`），运行/测试都要保证 `backend` 在 `sys.path`。
- 流水线编排在 `services/pipeline.py`，顺序 detect → ocr → translate → inpaint → render。引擎可插拔，统一通过 `services/engines/factory.py` 的 `get_engine(type)` 获取（lru_cache 缓存）。
- OCR 引擎自带检测（`supports_detection=True`）时，`pipeline.py` 跳过独立 detector，直接用 OCR 检测+识别。

## 易踩的坑（改动引擎时注意）

- **OCR（PaddleOCR 3.x）**：用 `ocr.predict()` 而非旧版 `.ocr()`；结果字段是 `rec_texts` / `rec_scores` / `rec_polys`。创建实例的 4 个参数不能删：`enable_mkldnn=False`、`use_doc_orientation_classify=False`、`use_doc_unwarping=False`、`use_textline_orientation=True`，否则漫画竖排文字误检。竖排靠把原图旋转 90° 再检测、坐标映射回原图（`ocr.py:_detect_all`）。
- **翻译**：Google 免费接口必须用 `client=dict-chrome-ex`（`gtx` 会被限流 429）。回退链 google → mymemory → 仅词典替换（`translator.py:SmartTranslator`）。
- **渲染字体**：Windows 用 `C:/Windows/Fonts/msyh.ttc`，候选列表在 `renderer.py:FONT_CANDIDATES`；横排字号基准取检测框高度中位数（`_unified_height`），竖排逐字垂直渲染。
- **图像修复**：双引擎。默认 `cv`（无模型）：`mask.py` 用 OCR 多边形（`region.poly`）+ Otsu 自适应阈值生成精确笔画掩膜（缓存到 `region.mask`），`inpainter.py` 对掩膜按行重采左右条带背景填充、边界环用 `cv2.INPAINT_TELEA` 羽化。可选 `lama`（神经网络，效果是 manga-image-translator 质量级）：需在虚拟环境装 `backend/requirements-inpaint.txt`（torch），权重自动搜 `backend/data/models/lama_large_512px.ckpt` 或本机 manga-image-translator 路径；配置 `MANGA_INPAINTER_BACKEND` / `MANGA_LAMA_MODEL_PATH` / `MANGA_LAMA_INPAINT_SIZE` / `MANGA_INPAINT_DEVICE`。LaMa 模型结构 vendor 在 `lama_model.py`（仅推理所需 FFC/LamaFourier，训练用判别器/MPE 已省略）。

## 约定

- Git 远程 `origin = https://github.com/WenQ-byte/manga-transtor.git`，分支 `master`（非 main）。提交后 `git push` 同步。
- 代码注释/UI/文档用中文；代码无注释（遵循现有风格，除非被要求）。
- `opencode.json` 加载 superpowers 插件；另有自定义 skill `.opencode/skills/taste-skill/`。
- README 的架构图写 `docs/prds/`，实际 PRD 文档在 `docs/` 平铺（`project-initiation-report.md`、`research-report.md` 等），路径已过时，勿据此找文件。
