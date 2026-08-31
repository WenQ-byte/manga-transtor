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
  当前 62 项测试；测试会设 `MANGA_DATA_DIR` 到临时目录；`test_pipeline_runs` 走真实 OCR 路径，若已装 AI 依赖会加载 PaddleOCR 模型、较慢。
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
- 流水线编排在 `services/pipeline.py`，顺序 detect → ocr → **inpaint → bubble grouping → translate** → quality assessment → render。分组在擦除后的干净图上寻找气泡内部，同时用原图长轮廓否决跨气泡合并；翻译按气泡整块进行。引擎可插拔，统一通过 `services/engines/factory.py` 的 `get_engine(type)` 获取（lru_cache 缓存）。
- OCR 引擎自带检测（`supports_detection=True`）时，`pipeline.py` 跳过独立 detector，直接用 OCR 检测+识别。
- OCR 翻译/擦除阈值由 `pipeline.py:ocr_thresholds` 分离：MIT48 路线翻译阈值 0.20、擦除阈值 0.0（检测框和掩膜通常比字符概率可靠）；Paddle/CV 路线保持 0.50/0.50。非空且达到翻译阈值的 region 参与翻译，其余有掩膜的 MIT region 仍可只参与擦除。
- **非气泡文字判定**（`bubble.py:classify_non_bubble`）：泛洪 bbox 为细长横条（w/h≥8 且高≤4%页高）或跨页宽横带（方向横且宽≥50%页宽且高≤5%页高）→ 判为刊头/拟声词等非气泡，标 `_no_erase`；`pipeline.py:drop_non_bubble_regions` 移出工作集（不擦除、不翻译、不渲染，原文保留），renderer 同跳过（双保险）。默认开启，不依赖旧 `is_ignore` 启发式（后者仅 config≥1 时 opt-in，且对贴字/彩色背景不可靠）。
- **掩膜整块化**：`mask.py` 优先按文本多边形整块填充（`fillPoly` + 膨胀），**并集 Otsu 笔画候选 + 7×7 膨胀**（poly 常偏紧、pad 缩进会漏掉笔画外缘 → 原文灰影残留）；bbox 以 box∩poly 并集为界。无 poly 才回退 Otsu 笔画。**注音扩展 `_add_furigana_margin`**：竖排汉字旁的小号 furigana 检测器常漏检 → 沿主字 bbox 外扩 16px 拾取带内小连通字形成分并擦除（`FURIGANA_MARGIN` 可调）。非气泡文字（刊头/拟声词）打 `_no_erase` 标记跳过擦除并整体移出翻译/渲染（原文保留）。
- 渲染防丢字：`renderer.py` 逐组统计「文字像素 ∩ 气泡掩膜」覆盖率，<50% 回退矩形裁剪（防泛洪掩膜不可信时整段文字被裁掉）。
- **分组防链式吞并**：`bubble.py:_balloon_ok` 要求合并后包围盒面积 ≤ 1.6×(两框面积和)——页眉横条/拟声词等细长区域的泛洪 bbox 若被近邻合并链式吞并整页会爆炸（跨页杂志实测 17 组并成 1 组导致整页巨字），膨胀校验阻断；渲染端 `_bubble_geometry` 另加绝对上限：泛洪结果不得超出 `group_bounds` 边长 15%，超限回退锚定矩形。
- **原图边界否决**：`group_regions_by_bubble(..., boundary_bgr=原图)` 在清理图上泛洪，但所有可靠掩膜合并和二次合并都要经过原图分隔线校验。两个文字组之间若存在跨越至少 45% 走廊的长黑轮廓则禁止合并；贴合/重叠的同气泡文字列视为开放通路，避免被较远轮廓误拆。JoJo 双页回归由错误 11 组（目标 1 组）修正为实际 13 组（目标 3 组）。
- **气泡几何回退分级**（`renderer.py:_bubble_geometry`）：一级可靠气泡掩膜（泛洪过全部可信度校验）→ 二级有限安全扩展框（`_safe_expand_box`：以文本框锚点向四周渐进扩展，受边缘/纹理检查约束——Sobel 梯度带边缘占比超阈值即停，阻止穿过气泡轮廓/分镜线/人物高纹理；上限约 `SAFE_EXPAND_RATIO`=1.8 倍锚点宽高）→ 三级紧致文本框（锚点本身，最后兜底）→ 四级跳过该气泡。**紧致框是最后手段，不是掩膜失败时的默认方案**；掩膜失败时优先用扩展框避免译文挤在 1px 内边距里。
- 排版均衡：竖排整块 `_balance_columns` 按原文行序分配列；超长单行由 `_split_semantic_columns` 在容量约束内均衡切列，保护闭合标点和常见中文双字词。横排 `_wrap_text` 均衡断行。翻译提示词强调保持原文换行/句子结构和顺序。`_render_vertical_bubble_block` 整块按气泡分列、列组对称居中顶部对齐。

## MIT 引擎（移植自 manga-image-translator，GPL-3.0）

代码在 `backend/app/services/engines/mit/`（裁剪为仅推理），引擎包装在 `engines/detector.py:MangaDetector` 与 `engines/ocr.py:MIT48OCREngine`。**MIT 是默认检测/OCR**（`MANGA_OCR_BACKEND` 默认 `mit48+mangaocr` → detector 自动 `manga`，检测器默认 `ctd`；MIT 加载失败或权重缺失时引擎工厂回退 mit48 → PaddleOCR/CV）。**建议 2112+ 显存或纯 CPU**；模型权重在 `backend/data/models/mit/`（gitignore，不提交），可用 `MANGA_MIT_MODEL_DIR` 指向已下载的 MIT 仓库 `models/`。

- **检测**：`mit_detector=default` 用自实现 ResNet34 + DBNet（`mit/resnet34.py` 替代 torchvision 的 resnet34，避免版本匹配问题，权重名一致）；`ctd`（默认）用 `mit/ctd.py` + `ctd_utils/`（YOLOv5-s backbone + UNet 文本掩膜头 + DBNet 行头；GPU 用 `.pt`，CPU 用 `.pt.onnx` + cv2.dnn；YOLO 块/语言检测输出已弃用，照搬 MIT 行为只消费 mask+lines）。
- **识别**：`mit/ocr_48px.py`（ConvNeXt 骨干 + RoFormer/XPOS + beam search，`infer_beam_batch_tensor`），`mit/xpos.py`。逐字符预测前景/背景色，回填 `TextRegion.fg_color/bg_color`。可选 `mangaocr`/`mit48+mangaocr`（`mit/mocr.py`，HF `kha-white/manga-ocr-base`，风格化字体更强但无置信度）。**混排只把 mit48 完全没读出的空行交给 manga-ocr**（`_empty_quads_only`）：mit48 对风格化/小字常读对但置信度低（0.45~0.6），若按 `prob<阈值` 交给 manga-ocr 会把已读对的内容覆盖成含拉丁字母乱码（NS/SM/纽哈梅/古哈米）——故 mocr 仅"救空"。**竖排裁剪**：`get_transformed_region` 会把竖排旋转 90° CCW 供水平训练模型；mit48 依赖该旋转（batch 张量固定高=text_height），`MangaOcrWrapper` 则用 `ROTATE_90_CLOCKWISE` 转回竖直再读（manga-ocr 支持任意朝向，旋转过的横条会读成乱码）。小字（字号 < `MANGA_MIT_OCR_UPSCALE`）先 2x 放大再识别（`Mit48Ocr._region_image`）。
- **翻译分组**：pipeline 翻译前用 `bubble.py:group_regions_by_bubble` 按气泡泛洪分组，整块 `\n` 拼接一次翻译（更地道），整块译文存 `region.group_translated` 并尽力按行拆回 `region.translated`。
- **方向**：`Quadrilateral.direction`（横/竖）由 `sort_pnts` 判定，回填 `TextRegion.direction`；渲染竖排判定以气泡形状为主（`bh > bw*MANGA_RENDER_VERTICAL_MIN_RATIO` 且方向 v 占多，或高远超宽），方向仅作联席依据，避免矮气泡被误判强制竖排。
- **排版**：文本绘制到透明 overlay，按气泡泛洪掩膜裁剪合成（`bubble.bubble_with_mask`）——文字绝不越出气泡框；横排单行优先 + 孤字规避（`_select_horizontal_font`），竖排整块按气泡分列、列组对称居中顶部对齐（`_render_vertical_bubble_block`）。内边距 `MANGA_RENDER_PADDING`。
- **掩膜**：检测阶段直接把 raw_mask 裁剪成 0/255 patch 存入 `region.mask`（`{bbox, patch}`）；`mask.py:region_patch` 将缓存掩膜与 OCR 多边形/Otsu 补充掩膜对齐取并集，覆盖抗锯齿边缘与注音。竖排注音只向左右扩展、横排只向上下扩展，且忽略触碰扩展带外边界的组件，避免误擦气泡轮廓。非气泡文字由 `classify_non_bubble` 判定并移出管线，原文保留。
- **管线配合**：MIT 检测器的每个 textline 即一个 region（不停靠 textline_merge，因为 renderer 按 region 分组渲染多行/多列气泡）；`MANGA_BUBBLE_FILTER` 默认 auto，MIT 模式自动关闭白占比气泡过滤（对彩色/深色气泡误删）。翻译按气泡整块进行（见上），渲染优先用 `group_translated` 整块排版（横排二分字号填满、竖排整块分列重排）。

## 易踩的坑（改动引擎时注意）

- **OCR（PaddleOCR 3.x）**：用 `ocr.predict()` 而非旧版 `.ocr()`；结果字段是 `rec_texts` / `rec_scores` / `rec_polys`。创建实例的 4 个参数不能删：`enable_mkldnn=False`、`use_doc_orientation_classify=False`、`use_doc_unwarping=False`、`use_textline_orientation=True`，否则漫画竖排文字误检。竖排靠把原图旋转 90° 再检测、坐标映射回原图（`ocr.py:_detect_all`）。
- **翻译**：Google 免费接口必须用 `client=dict-chrome-ex`（`gtx` 会被限流 429）。回退链为配置后端 → 其他已配置高质量后端（deepseek/openai/deepl）→ google → mymemory → 仅词典/原文；实际后端写入 `last_backend_names`，失败写入 `last_failures`。批量后端置 `batch = True` 并提供 `translate_batch`：DeepL 是数组一次调用；DeepSeek 是页级上下文批量（整页全部段编号 `<i>...</i>` 合并一次请求，`_parse_segments` 解析回填）。`assess_translation_quality` 标记免费/原文回退、译文明显过短、日文残留和数字遗漏。词典按整词替换。
- **渲染字体**：Windows 用 `C:/Windows/Fonts/msyh.ttc`，候选列表在 `renderer.py:FONT_CANDIDATES`；横排字号基准取检测框高度中位数（`_unified_height`），竖排逐字垂直渲染（`VERTICAL_CHAR_RATIO=1.15` 字距、`VERTICAL_COL_USE_RATIO=0.82` 列宽占比防重叠、横排 `LINE_SPACING_RATIO=0.3` 中文行距需偏大）。
- **图像修复**：双引擎。默认 `cv`：平坦亮背景在 OCR 四边形安全区内按行重建，剩余掩膜用 TELEA，最后只在已知文字区域附近做保守残影二次检测。可选 `lama`：先处理平坦背景，再将相邻掩膜聚类为局部 crop 原尺寸推理；只有 crop 总面积超过页面 85% 才回退整页推理，避免整页缩到 1024 后小字生成灰影。权重与配置同 `MANGA_INPAINTER_BACKEND` / `MANGA_LAMA_MODEL_PATH` / `MANGA_LAMA_INPAINT_SIZE` / `MANGA_INPAINT_DEVICE`。

## 约定

- Git 远程 `origin = https://github.com/WenQ-byte/manga-transtor.git`，分支 `master`（非 main）。提交后 `git push` 同步。
- **push 需走代理**：本机已 `git config --global http.proxy https.proxy = http://127.0.0.1:7897`（Clash Verge 规则模式）。git.exe 不读系统代理环境变量，若全局配置被清空则直接 push 会 `Failed to connect to github.com port 443`，需重新配置代理。
- 代码注释/UI/文档用中文；代码无注释（遵循现有风格，除非被要求）。
- `opencode.json` 加载 superpowers 插件；另有自定义 skill `.opencode/skills/taste-skill/`。
- 项目文档在 `docs/` 平铺（`project-initiation-report.md`、`research-report.md` 等，另有命名含日期的 .txt 竞品分析）。README 的架构说明以实际代码为准（此前 README 里 `docs/prds/` 路径已过时，勿据此找文件）。
