# 漫译 · 漫画多语言智能翻译系统

面向中文、日语和英语漫画的三语互译 Web 应用。系统完成文本检测、OCR、原文擦除、气泡分组、上下文翻译和多语言排版，并尽量保持原漫画的分镜、气泡轮廓与阅读顺序。

当前版本为本地可运行 MVP：FastAPI 后端 + React/Vite 前端。默认检测与 OCR 路线移植自 `manga-image-translator`，相关代码遵循 GPL-3.0，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 当前进度

截至 2026-08-31：

- 支持中文、日语、英语之间全部六种翻译方向，单图与批量流程共用同一套语言配置。
- 支持自动检测源语言，默认行为为“自动检测 → 中文”。
- 支持最多 100 张图片批量创建独立翻译任务、汇总进度，并将成功结果和清单打包为 ZIP。
- 默认使用 CTD 检测器与 `mit48+mangaocr` 混合 OCR，PaddleOCR 作为回退方案。
- 支持 CV 与 LaMa 两种修复引擎；LaMa 采用局部原尺寸推理，减少整页缩放造成的灰色残影。
- 支持 DeepSeek、OpenAI、DeepL、Google 和 MyMemory，并提供自动回退与质量告警。
- 中文和日语支持横竖排自适应；英语目标采用安全横排并优先在词间换行。
- 气泡分组同时使用清理图内部区域和原图轮廓，避免修复后轮廓变弱导致相邻气泡误合并。
- 当前自动化测试共 108 项，全部通过；JoJo 双页回归图识别为 13 个气泡，目标相邻区域稳定拆分为 3 个气泡。

## 主要能力

- 上传 JPG、JPEG、PNG、WebP、BMP，默认单文件上限 10 MB；批量默认最多 100 张、总计 500 MB。
- 每张批量图片复用独立单图流水线，支持总体/单图进度、部分失败和 ZIP 汇总导出。
- 批量子任务按顺序进入共享模型流水线，避免 CTD、OCR 和翻译器实例并发造成检测框、掩膜或识别结果污染。
- CTD/DBNet 漫画文本检测，支持复杂分镜、竖排文字和网点背景。
- MIT 48px OCR 与 manga-ocr 混合识别；manga-ocr 只补救 MIT 完全未识别的空行，避免覆盖低置信度但正确的结果。
- MIT OCR 的翻译阈值与擦除阈值分离：可读低置信度文本继续翻译，有检测掩膜的低置信度文字仍可擦除。
- 非气泡刊头、拟声词和高比例拉丁标签可保留原样，不参与擦除、翻译和渲染。
- OCR 多边形、逐像素检测掩膜、Otsu 笔画与注音扩展联合生成擦除掩膜。
- 白色气泡使用受限局部背景重建，复杂纹理交给 CV TELEA 或 LaMa；修复后执行保守残影检测。
- 气泡级上下文翻译、页级 DeepSeek 批量翻译、专有名词整词替换。
- 记录请求语言、自动识别结果、实际翻译后端与回退原因，并按语言方向检查原文直出、异常过短、语言残留和数字遗漏。
- 中文横排均衡换行；竖排按原文阅读顺序分列，并避免拆开常见双字词和闭合标点。
- 气泡掩膜、有限安全扩展框、紧致文本框、跳过渲染四级回退，降低文字越界和覆盖人物的风险。
- 实时步骤进度、网页预览、结果下载和图形化专有名词管理。

## 技术栈

| 层级 | 技术 |
|---|---|
| 后端 | Python、FastAPI、Pydantic Settings、Pillow、OpenCV、httpx |
| 前端 | React 18、Vite 6、Tailwind CSS v4、Framer Motion、GSAP、Lucide |
| 检测/OCR | CTD、DBNet/ResNet34、MIT 48px OCR、manga-ocr、PaddleOCR 3.x |
| 图像修复 | OpenCV TELEA、局部背景重建、LaMa FFC |
| 翻译 | DeepSeek、OpenAI、DeepL、Google、MyMemory、专有名词词典 |
| 存储 | SQLite、文件系统 |

## 快速开始

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
pip install -r backend/requirements-ai.txt
pip install -r backend/requirements-inpaint.txt
pip install -r backend/requirements-mit.txt

cd frontend
npm.cmd install
cd ..
```

依赖分为四档：

- `requirements.txt`：FastAPI、Pillow、OpenCV、httpx 等核心依赖；只有这一档时可运行 Demo 流程。
- `requirements-ai.txt`：PaddleOCR 3.x。
- `requirements-inpaint.txt`：PyTorch 与 LaMa 修复。
- `requirements-mit.txt`：MIT 检测/OCR、manga-ocr 及几何依赖；使用默认 OCR 路线时需要安装。

MIT 与 LaMa 权重不提交到 Git。MIT 权重默认放在 `backend/data/models/mit/`，LaMa 权重默认放在 `backend/data/models/lama_large_512px.ckpt`，也可通过环境变量指定其他位置。

### 2. 配置

```powershell
Copy-Item .env.example .env
```

建议至少配置一个高质量翻译后端：

```dotenv
MANGA_TRANSLATOR_BACKEND=deepseek
MANGA_DEEPSEEK_API_KEY=sk-xxxx

MANGA_OCR_BACKEND=mit48+mangaocr
MANGA_MIT_DETECTOR=ctd
MANGA_MIT_DEVICE=auto

MANGA_INPAINTER_BACKEND=lama
MANGA_INPAINT_DEVICE=cpu
```

如果没有 `.env`，代码级翻译默认值为 Google；`.env.example` 推荐 DeepSeek。未配置或不可用的高质量后端会自动回退到 Google、MyMemory，最后仅应用词典并保留原文。

语言默认配置如下：

```dotenv
MANGA_DEFAULT_SOURCE_LANG=auto
MANGA_DEFAULT_TARGET_LANG=zh
MANGA_AUTO_SOURCE_FALLBACK=ja
```

自动检测基于整页 OCR 文本：出现假名时优先判为日语，拉丁字母占主导时判为英语，不含假名的汉字文本判为中文。空文本、纯数字和信息不足的短文本使用 `MANGA_AUTO_SOURCE_FALLBACK`，原因与置信度会写入任务元数据。

### 3. 启动

推荐双击 `start.bat`，或运行：

```powershell
.\.venv\Scripts\python.exe start.py
```

手动启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

手动启动前端：

```powershell
cd frontend
npm.cmd run dev
```

- 前端开发地址：<http://localhost:5173>
- 后端地址：<http://localhost:8000>
- OpenAPI 文档：<http://localhost:8000/docs>

Windows 上建议使用 `npm.cmd`，避免 PowerShell 执行策略阻止 `npm.ps1`。

### Docker

```powershell
docker compose up -d --build
```

Docker 镜像是否包含全部 AI 模型取决于本地权重与构建配置；首次部署前请确认模型目录和显存/内存需求。

## 核心配置

| 环境变量 | 说明 | 代码默认值 |
|---|---|---|
| `MANGA_PIPELINE_MODE` | `real` / `demo` | `real` |
| `MANGA_OCR_BACKEND` | `mit48+mangaocr` / `mit48` / `mangaocr` / `paddle` | `mit48+mangaocr` |
| `MANGA_DETECTOR_BACKEND` | 空值自动选择，或 `cv` / `manga` | 空值 |
| `MANGA_MIT_DETECTOR` | `ctd` / `default` | `ctd` |
| `MANGA_MIT_DEVICE` | `auto` / `cpu` / `cuda` / `mps` | `auto` |
| `MANGA_MIT_MODEL_DIR` | MIT 模型目录 | `backend/data/models/mit` |
| `MANGA_MIT_OCR_UPSCALE` | 小于该字号的文本先放大识别 | `16` |
| `MANGA_BUBBLE_FILTER` | `auto` / `on` / `off` | `auto` |
| `MANGA_TRANSLATOR_BACKEND` | `deepseek` / `openai` / `deepl` / `google` / `mymemory` | `google` |
| `MANGA_DEEPSEEK_MODEL` | 日语等非英语翻译使用的 DeepSeek 模型 | `deepseek-v4-flash` |
| `MANGA_DEEPSEEK_ENGLISH_MODEL` | 英语→中文使用的 DeepSeek 模型 | `deepseek-v4-flash` |
| `MANGA_INPAINTER_BACKEND` | `cv` / `lama` | `cv` |
| `MANGA_LAMA_MODEL_PATH` | LaMa 权重路径 | 自动搜索 |
| `MANGA_LAMA_INPAINT_SIZE` | 单个 LaMa 推理区域的最长边上限 | `1024` |
| `MANGA_INPAINT_DEVICE` | `cpu` / `cuda` | `cpu` |
| `MANGA_RENDER_PADDING` | 气泡内边距比例 | `0.12` |
| `MANGA_MAX_UPLOAD_MB` | 上传大小上限 | `10` |
| `MANGA_BATCH_MAX_FILES` | 单批图片数量上限 | `100` |
| `MANGA_BATCH_MAX_TOTAL_MB` | 单批总大小上限（MB） | `500` |

完整配置见 [.env.example](.env.example) 和 [backend/app/config.py](backend/app/config.py)。

## 流水线

```text
上传图片
  → 文本检测
  → OCR 与置信度分流
  → 非气泡文字/拉丁标签保护
  → 原文擦除与背景修复
  → 气泡分组
      · 清理图：寻找无文字遮挡的气泡内部
      · 原图：检测分隔轮廓，否决跨气泡合并
  → 气泡级/页级上下文翻译
  → 翻译质量检查
  → 横排/竖排中文渲染
  → 结果下载
```

修复发生在翻译之前，因此气泡分组可以在没有原文字笔画干扰的图上进行；同时保留原图作为边界证据，避免修复削弱轮廓后相邻气泡串联。

## 项目结构

```text
backend/
  app/
    api/                    REST API
    services/
      pipeline.py           流水线编排与质量信息
      engines/
        factory.py          引擎工厂与缓存
        detector.py         CV/MIT 检测器包装
        ocr.py              Paddle/MIT/manga-ocr 包装
        translator.py       翻译后端、批量请求与回退链
        bubble.py           气泡过滤、分组与原图边界校验
        mask.py             OCR 多边形、笔画与注音掩膜
        inpainter.py        CV 修复与残影清理
        lama.py             LaMa 局部推理
        renderer.py         气泡几何与中文排版
        mit/                manga-image-translator 推理代码
    storage/                SQLite 与文件存储
frontend/                   React/Vite 前端
docs/                       项目宪章、研究报告和需求资料
test_image/                 本地回归漫画，不纳入版本库
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/translate` | 上传图片并创建翻译任务 |
| `POST` | `/api/translate/batch` | 批量上传并创建多个独立单图任务 |
| `POST` | `/api/translate/batch/status` | 汇总多个子任务的状态与平均进度 |
| `POST` | `/api/translate/batch/zip` | 导出成功图片、清单与失败说明 ZIP |
| `GET` | `/api/translate/{task_id}/status` | 查询任务进度 |
| `GET` | `/api/translate/{task_id}/result` | 下载翻译结果 |
| `DELETE` | `/api/translate/{task_id}` | 删除任务 |

单图接口通过查询参数、批量接口通过表单字段接收 `source_lang` 与 `target_lang`。`source_lang` 支持 `auto`、`zh`、`ja`、`en`，`target_lang` 支持 `zh`、`ja`、`en`；显式选择相同语言会返回 400。旧客户端不传参数时按“自动检测 → 中文”运行。批量中的全部图片使用同一组语言配置，每张图片的实际识别语言、后端和回退信息分别记录。

DeepSeek 与 OpenAI 使用按目标语言生成的漫画提示词；DeepSeek继续提供页级编号上下文和解析失败后的逐条回退。DeepL、Google、MyMemory 通过统一映射层转换语言代码，不支持的代码会失败并进入既有回退链。MIT48 与 manga-ocr 主要针对日语漫画优化；中文源图和复杂英文源图若识别不稳定，建议显式选择语言并使用 PaddleOCR。
| `GET` / `POST` | `/api/glossary` | 查询或新增词条 |
| `PUT` / `DELETE` | `/api/glossary/{item_id}` | 修改或删除词条 |
| `POST` | `/api/glossary/import` | JSON 批量导入词典 |
| `GET` | `/api/files/{filename}` | 获取结果文件 |

## 测试

```powershell
.\.venv\Scripts\python.exe backend\tests\test_app.py
```

测试使用标准库 `unittest`，不是 pytest。部分测试会加载真实 PaddleOCR 模型，因此首次执行可能较慢。

当前覆盖重点包括单图/批量 API、ZIP 安全导出与流水线，以及 OCR 适配、掩膜与残影清理、LaMa 局部推理、气泡分组与原图边界校验、气泡几何回退、中文布局、翻译回退和质量告警。

## 当前限制

- 批量功能是对独立单图任务的编排与汇总，尚无可恢复的章节级持久队列、暂停/续跑和批次历史管理。
- 暂无用户账户、历史任务同步、人工校对编辑器和团队协作。
- MIT、manga-ocr 与 LaMa 模型体积较大，首次准备环境需要额外下载时间和磁盘空间。
- CPU 可运行，但 CTD、manga-ocr 和 LaMa 的整页处理速度明显慢于 GPU。
- 免费翻译后端仅作为兜底；正式使用建议配置 DeepSeek、OpenAI 或其他高质量模型。
- 极端艺术字、开放式气泡、深色气泡或文字压线仍需通过真实漫画持续回归。

## 文档

- [项目宪章与进度基线](docs/project-initiation-report.md)
- [研究报告](docs/research-report.md)
- [第三方代码与许可证](THIRD_PARTY.md)

## License

本项目采用 GPL-3.0。第三方模型、代码和服务可能有独立许可或使用条款，发布和商业使用前请逐项核对。
