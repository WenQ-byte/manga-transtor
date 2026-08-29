# AGENTS.md

漫画多语言智能翻译系统（日语/英语 → 中文）。FastAPI 后端 + React/Vite 前端（Tailwind v4 + Framer Motion + GSAP + Lucide） + PaddleOCR。

## 运行 / 测试命令

- 一键启动（推荐）：双击 `start.bat`，或 `.venv\Scripts\python.exe start.py`，交互菜单选后端/前端/前后端。`start.py` 是中文菜单 + subprocess 启动；`start.bat` 仅纯 ASCII 外壳（避免 bat 中文编码问题）。
- 手动启动后端（必须加 `--app-dir backend`，模块是 `app.main:app`）：
  ```
  .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
  ```
- 前端：`cd frontend; npm run build`（生产构建）或 `npm run dev`（Vite 开发模式，端口 5173）。**Windows 上 npm 用 `npm.cmd`，`npm.ps1` 被执行策略禁止。**
- 测试（stdlib `unittest`，非 pytest）：
  ```
  .venv\Scripts\python.exe backend\tests\test_app.py
  ```
  测试会设 `MANGA_DATA_DIR` 到临时目录；`test_pipeline_runs` 走真实 OCR 路径，若已装 AI 依赖会加载 PaddleOCR 模型、较慢。
- 无 linter / typecheck / formatter 配置，不需要跑。

## 依赖四档

- `backend/requirements.txt`：核心（FastAPI/Pillow/opencv/httpx）。仅装这个时 OCR 走 Demo 模式（模拟识别）。
- `backend/requirements-ai.txt`：`paddlepaddle==3.3.1` + `paddleocr==3.7.0`，装了才启用真实 OCR。
- `backend/requirements-inpaint.txt`：`torch>=2.2`，装了才能启用 LaMa 神经修复。
- `backend/requirements-mit.txt`：`torchvision`（可选，自实现 resnet34 也可替代）、`einops`、`py3langid`、`pyclipper`、`shapely`、`networkx`，装了才能启用 MIT 检测/OCR 引擎。
- 首次真实 OCR 会自动下载模型到 `~/.paddlex/official_models/`；可用 `PADDLEOCR_HOME` 指向 `backend/data/models`。

## 配置

- pydantic-settings，环境变量前缀 `MANGA_`，读 `.env`（见 `backend/app/config.py`）。如 `MANGA_PIPELINE_MODE`（real/demo）、`MANGA_TRANSLATOR_BACKEND`（deepseek/google/deepl/openai/mymemory）、`MANGA_MAX_UPLOAD_MB`、`MANGA_INPAINTER_BACKEND`（cv/lama）、`MANGA_OCR_BACKEND`（mit48/mangaocr/mit48+mangaocr/paddle）、`MANGA_DETECTOR_BACKEND`（空/cv/manga）、`MANGA_MIT_DETECTOR`（default/ctd）、`MANGA_MIT_MODEL_DIR`/`MANGA_MIT_FALLBACK_DIR`、`MANGA_MIT_DEVICE`（cpu/cuda/mps/auto）、`MANGA_MIT_OCR_UPSCALE`/`MANGA_MIT_OCR_MIX_THRESHOLD`、`MANGA_BUBBLE_FILTER`（auto/on/off）。
- `.env` 已 gitignore；`.env.example` 展示全部可配项。DeepL 免费版 auth key 以 `:fx` 结尾。

## 架构要点（非文件名能看出的）

- 应用代码在 `backend/app`（不是 `backend`），运行/测试都要保证 `backend` 在 `sys.path`。
- 流水线编排在 `services/pipeline.py`，顺序 detect → ocr → translate → inpaint → render。引擎可插拔，统一通过 `services/engines/factory.py` 的 `get_engine(type)` 获取（lru_cache 缓存）。
- OCR 引擎自带检测（`supports_detection=True`，如 PaddleOCR）时，`pipeline.py` 跳过独立 detector，直接用 OCR 检测+识别。
- **默认组合**：`MANGA_OCR_BACKEND=mit48+mangaocr`（默认）→ `supports_detection=False`，走 `detector=MangaDetector(ctd) → 混合识别` 流程；显式设 `MANGA_OCR_BACKEND=paddle` 才回到 PaddleOCR 一体化检测+识别。
- OCR 结果按 `confidence >= 0.5` 且非空文本过滤噪声（`pipeline.py` 中 `regions = [r for r in regions if r.confidence >= 0.5 and r.text.strip()]`）。

## MIT 引擎（移植自 manga-image-translator，GPL-3.0）

代码在 `backend/app/services/engines/mit/`（裁剪为仅推理），引擎包装在 `engines/detector.py:MangaDetector` 与 `engines/ocr.py:MIT48OCREngine`。**MIT 是默认检测/OCR**（`MANGA_OCR_BACKEND` 默认 `mit48+mangaocr` → detector 自动 `manga`，检测器默认 `ctd`；MIT 加载失败或权重缺失时引擎工厂回退 mit48 → PaddleOCR/CV）。**建议 2112+ 显存或纯 CPU**；模型权重在 `backend/data/models/mit/`（gitignore，不提交），可用 `MANGA_MIT_MODEL_DIR` 指向已下载的 MIT 仓库 `models/`。

- **检测**：`mit_detector=default` 用自实现 ResNet34 + DBNet（`mit/resnet34.py` 替代 torchvision 的 resnet34，避免版本匹配问题，权重名一致）；`ctd`（默认）用 `mit/ctd.py` + `ctd_utils/`（YOLOv5-s backbone + UNet 文本掩膜头 + DBNet 行头；GPU 用 `.pt`，CPU 用 `.pt.onnx` + cv2.dnn；YOLO 块/语言检测输出已弃用，照搬 MIT 行为只消费 mask+lines）。
- **识别**：`mit/ocr_48px.py`（ConvNeXt 骨干 + RoFormer/XPOS + beam search，`infer_beam_batch_tensor`），`mit/xpos.py`。逐字符预测前景/背景色，回填 `TextRegion.fg_color/bg_color`。可选 `mangaocr`/`mit48+mangaocr`（`mit/mocr.py`，HF `kha-white/manga-ocr-base`，风格化字体更强但无置信度，混合模式按 `MANGA_MIT_OCR_MIX_THRESHOLD` 兜底）。小字（字号 < `MANGA_MIT_OCR_UPSCALE`）先 2x 放大再识别（`Mit48Ocr._region_image`）。
- **翻译分组**：pipeline 翻译前用 `bubble.py:group_regions_by_bubble` 按气泡泛洪分组，整块 `\n` 拼接一次翻译（更地道），整块译文存 `region.group_translated` 并尽力按行拆回 `region.translated`。
- **方向**：`Quadrilateral.direction`（横/竖）由 `sort_pnts` 判定，回填 `TextRegion.direction`；渲染仍按气泡宽高比决定横竖排（renderer 未消费 direction，留作后续）。
- **掩膜**：检测阶段直接把 raw_mask 裁剪成 0/255 patch 存入 `region.mask`（`{bbox, patch}`），`mask.py:build_full_mask` 已改为**缓存优先**（否则会覆盖 MIT 预填充）。`MANGA_MIT_IGNORE_BUBBLE`（1-50）开启时，非气泡区域（拟声词等）不生成掩膜、原文保留。
- **管线配合**：MIT 检测器的每个 textline 即一个 region（不停靠 textline_merge，因为 renderer 按 region 分组渲染多行/多列气泡）；`MANGA_BUBBLE_FILTER` 默认 auto，MIT 模式自动关闭白占比气泡过滤（对彩色/深色气泡误删）。翻译按气泡整块进行（见上），渲染优先用 `group_translated` 整块排版（横排二分字号填满、竖排整块分列重排）。

## 易踩的坑（改动引擎时注意）

- **OCR（PaddleOCR 3.x）**：用 `ocr.predict()` 而非旧版 `.ocr()`；结果字段是 `rec_texts` / `rec_scores` / `rec_polys`。创建实例的 4 个参数不能删：`enable_mkldnn=False`、`use_doc_orientation_classify=False`、`use_doc_unwarping=False`、`use_textline_orientation=True`，否则漫画竖排文字误检。竖排靠把原图旋转 90° 再检测、坐标映射回原图（`ocr.py:_detect_all`）。
- **翻译**：Google 免费接口必须用 `client=dict-chrome-ex`（`gtx` 会被限流 429）。回退链 google → mymemory → 仅词典替换（`translator.py:SmartTranslator`）。DeepL 支持批量：DeepLTranslator 置 `batch = True` 并提供 `translate_batch`；新增批量后端要仿照此模式（`SmartTranslator.translate_batch` 用 `getattr(backend, "batch", False)` 分派）。词典按整词替换（`\w` 边界 regex 在 `_apply_glossary`，含 CJK 也算单词字符）。
- **渲染字体**：Windows 用 `C:/Windows/Fonts/msyh.ttc`，候选列表在 `renderer.py:FONT_CANDIDATES`；横排字号基准取检测框高度中位数（`_unified_height`），竖排逐字垂直渲染（`VERTICAL_CHAR_RATIO=1.15` 字距、`VERTICAL_COL_USE_RATIO=0.82` 列宽占比防重叠、横排 `LINE_SPACING_RATIO=0.3` 中文行距需偏大）。
- **图像修复**：双引擎。默认 `cv`（无模型）：`mask.py` 用 OCR 多边形（`region.poly`）+ Otsu 自适应阈值生成精确笔画掩膜（缓存到 `region.mask`），`inpainter.py` 对掩膜按行重采左右条带背景填充、边界环用 `cv2.INPAINT_TELEA` 羽化。可选 `lama`（神经网络，效果是 manga-image-translator 质量级）：需在虚拟环境装 `backend/requirements-inpaint.txt`（torch），权重自动搜 `backend/data/models/lama_large_512px.ckpt` 或本机 manga-image-translator 路径；配置 `MANGA_INPAINTER_BACKEND` / `MANGA_LAMA_MODEL_PATH` / `MANGA_LAMA_INPAINT_SIZE` / `MANGA_INPAINT_DEVICE`。LaMa 模型结构 vendor 在 `lama_model.py`（仅推理所需 FFC/LamaFourier，训练用判别器/MPE 已省略）。

## 约定

- Git 远程 `origin = https://github.com/WenQ-byte/manga-transtor.git`，分支 `master`（非 main）。提交后 `git push` 同步。
- **push 需走代理**：本机已 `git config --global http.proxy https.proxy = http://127.0.0.1:7897`（Clash Verge 规则模式）。git.exe 不读系统代理环境变量，若全局配置被清空则直接 push 会 `Failed to connect to github.com port 443`，需重新配置代理。
- 代码注释/UI/文档用中文；代码无注释（遵循现有风格，除非被要求）。
- `opencode.json` 加载 superpowers 插件；另有自定义 skill `.opencode/skills/taste-skill/`。
- 项目文档在 `docs/` 平铺（`project-initiation-report.md`、`research-report.md` 等，另有命名含日期的 .txt 竞品分析）。README 的架构说明以实际代码为准（此前 README 里 `docs/prds/` 路径已过时，勿据此找文件）。