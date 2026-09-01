# 项目状态交接

更新时间：2026-09-01  
当前分支：`master`  
最新提交：`f8b4380 feat: add chat-style translation task flow`  
远端：`origin/master` 已同步

## 项目定位

本项目是漫画多语言智能翻译系统，目标是把日语/英语漫画图片翻译成中文，并在原图上完成文字擦除、气泡分组、译文排版、结果预览和下载。

技术栈：

- 后端：FastAPI，核心代码在 `backend/app`
- 前端：React/Vite，Tailwind v4，Framer Motion，GSAP，Lucide
- OCR/检测：默认优先 MIT 路线，PaddleOCR 作为重要补充和回退
- 图像处理：OpenCV/Pillow，默认 CV 修复，可选 LaMa
- 测试：stdlib `unittest`，不是 pytest

## 当前仓库状态

已提交并推送的最新变更主要包括：

- 前端第一阶段 ChatGPT 式翻译任务界面已落地并推送。
- 主区域从传统表单结果改为任务流：每次提交生成一个批次，批次内每张图片独立展示状态、进度、预览、单张下载和重新翻译入口。
- 底部固定输入区支持选择多张图片、粘贴剪贴板图片、展示待提交图片，提交成功后清空输入区。
- 复用现有单图/批量翻译、批量状态、结果图片和 ZIP 下载接口；未新增后端接口，未修改检测、OCR、翻译或渲染算法。
- 批次轮询合并为单个前端定时器，支持刷新页面后继续查询已知批次中的任务状态。
- 多语言翻译流水线增强，覆盖日/英/中相关语言路径。
- PaddleOCR 设备配置和 GPU 优先、失败回退 CPU 的能力。
- OCR、气泡分组、掩膜、渲染、翻译质量评估等核心链路的稳定性改进。
- 后端 API/schema/config/task manager 适配新增流水线信息。
- 前端 `TranslatePanel` 展示补充。
- README、`.env.example`、三语互译说明和阶段升级路线图更新。
- 后端测试大幅扩充，当前本地通过 159 项测试。
- 日→中真实样例回归已修正两类漏识别：日文页面中“单个假名 + 汉字”的区域不再误判成中文；低置信短竖列会在安全范围向上补充上下文后复识别。
- 日文片假名与汉字混写的人名会作为疑似专名传入翻译提示，避免按字面拆译；爱心符号在翻译返回方框或遗漏时会保留。

当前 `git status` 仍有已修改但未提交的后端/测试文件，来自本次前端提交之前的工作区状态，未纳入 `f8b4380`：

- `backend/app/services/engines/mit/mocr.py`
- `backend/app/services/engines/ocr.py`
- `backend/app/services/engines/renderer.py`
- `backend/app/services/engines/translator.py`
- `backend/app/services/pipeline.py`
- `backend/tests/test_app.py`

当前 `git status` 仍有未跟踪目录，未纳入提交：

- `.opencode/skills/ai-tutor/`
- `.opencode/skills/elon-musk-skill-main/`
- `.opencode/skills/guizang-ppt/`
- `.opencode/skills/learning-report-uploader/`
- `.opencode/skills/llm-wiki/`
- `.opencode/skills/zhangxuefeng-skill-main/`
- `test_image_chinese/`
- `test_image_english/`
- `test_image_japanese/`

这些看起来是个人技能包和测试截图/样例素材。后续如果要纳入版本库，需要用户明确确认；否则保持未跟踪即可。

## 已验证状态

最近一次测试命令：

```powershell
.\.venv\Scripts\python.exe backend\tests\test_app.py
```

结果：

```text
Ran 159 tests in 28.049s
OK
```

测试过程中加载了 PaddleOCR 缓存模型，并出现过 GPU 回退 CPU 的日志；这属于当前设备/显存/Paddle 环境下的预期回退路径。

最近一次前端构建命令：

```powershell
cd frontend
npm.cmd run build
```

结果：

```text
vite v6.4.3
✓ 2005 modules transformed
✓ built in 11.51s
```

该构建验证了第一阶段任务界面改造后的生产包可正常生成。

## 运行方式

推荐一键启动：

```powershell
.\.venv\Scripts\python.exe start.py
```

手动启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

前端开发：

```powershell
cd frontend
npm.cmd run dev
```

前端构建：

```powershell
cd frontend
npm.cmd run build
```

Windows 上使用 `npm.cmd`，不要直接用 `npm.ps1`。

## 关键配置

配置通过 `.env` 和 `MANGA_` 前缀环境变量读取，主要入口在 `backend/app/config.py`。

常用配置项：

- `MANGA_PIPELINE_MODE`：`real` 或 `demo`
- `MANGA_TRANSLATOR_BACKEND`：`deepseek`、`google`、`deepl`、`openai`、`mymemory`
- `MANGA_OCR_BACKEND`：`mit48`、`mangaocr`、`mit48+mangaocr`、`paddle`
- `MANGA_DETECTOR_BACKEND`：空、`cv`、`manga`
- `MANGA_MIT_DETECTOR`：`default` 或 `ctd`
- `MANGA_MIT_DEVICE`：`cpu`、`cuda`、`mps`、`auto`
- `MANGA_PADDLE_DEVICE`：例如 `gpu:0` 或 `cpu`
- `MANGA_INPAINTER_BACKEND`：`cv` 或 `lama`
- `MANGA_BUBBLE_FILTER`：`auto`、`on`、`off`

`.env` 已被 gitignore，不要提交真实密钥。`.env.example` 用于展示可配置项。

## 核心架构提醒

流水线主要在 `backend/app/services/pipeline.py`：

```text
detect -> ocr -> inpaint -> bubble grouping -> translate -> quality assessment -> render
```

需要特别注意：

- OCR 引擎如果 `supports_detection=True`，pipeline 会跳过独立 detector。
- 批量翻译只是并发编排任务，不并发模型推理。
- `TranslationTaskManager` 的单 worker 和 `_pipeline_lock` 必须保留。
- 检测、OCR、翻译等模型引擎通过 `get_engine(type)` 全局缓存，不能随意并发进入同一条模型流水线。
- MIT 路线默认检测/OCR，权重目录在 `backend/data/models/mit/`，该目录不提交。
- PaddleOCR 3.x 使用 `ocr.predict()`，不是旧版 `.ocr()`。
- Google 免费接口需要 `client=dict-chrome-ex`，不要改回容易 429 的 `gtx`。

## 当前优先级

后续主线不建议继续堆零散模型功能，而是优先把项目做成更完整、可演示、可继续扩展的产品形态。

第一优先级：ChatGPT 式翻译任务界面（已完成第一阶段）

- 已实现底部固定输入区，用于上传或粘贴多张图片。
- 已实现主区域对话/任务流形式，每次提交显示一个用户批次和对应系统处理结果。
- 已实现每张图片独立展示等待中、处理中、完成、失败状态；处理中会显示当前阶段和进度。
- 已实现单张完成后立即预览，不需要等待整批结束。
- 已实现单张重新翻译、单张下载。
- 已保留批量 ZIP 下载，并且只在批次内存在可下载结果时显示。
- 已处理单张失败隔离：失败图片显示错误信息，不阻塞其他图片继续轮询、预览和下载。
- 已做刷新续查的最低限度支持：前端把批次和任务 ID 存到 `localStorage`，刷新后可继续查询状态和显示结果；刷新后无法保留原始上传文件二进制，因此已完成图片仍可预览/下载，重新翻译需要用户重新选择图片。

下一优先级：侧边栏信息架构和专有名词库

- “翻译任务”和“专有名词库”作为同级入口。
- 点击“专有名词库”后，主区域切换为词库编辑界面。
- 词库编辑优先做 CRUD、搜索、按语言方向筛选。
- 不要把词库长期塞在翻译表单里。

后续优先级：性能与英文 OCR（已完成）

- 优先完善 PaddleOCR 的 GPU 配置、实际设备记录、失败回退原因记录。
- 不要承诺固定加速倍数。
- 不要通过并发整条流水线来换速度。
- 后续再考虑减少低置信度重复候选、懒加载语言模型、减少重复读图。

后续优先级：翻译质量继续细化

- 优先日译中、英译中。
- 中/日/英六路径保留框架能力即可，不要为了六路径全量打磨拖慢体验闭环。
- 专有名词库优先服务日译中和英译中。
- 自定义翻译风格等功能适合在任务界面稳定后再接入。

## 暂缓事项

这些方向有价值，但不建议压进第一阶段：

- 完整账号体系
- 团队协作
- 全量历史任务持久化和复杂检索
- 文件夹上传的复杂递归规则
- 所有六条翻译路径的深度质量打磨
- 多任务并发模型推理

## 给下一个接手者

1. 先读 `AGENTS.md` 和 `docs/阶段升级路线图.md`。
2. 第一阶段任务界面已经在 `frontend/src/components/TranslatePanel.jsx`、`frontend/src/App.jsx`、`frontend/src/index.css` 中实现；接手 UI 前先跑通现有上传、粘贴、单张完成预览、失败隔离、单张下载、重新翻译和 ZIP 下载流程。
3. 做 UI 时保持中文文案，沿用当前 React/Vite/Tailwind/Lucide 风格。
4. 做后端时优先保持现有 pipeline 边界，不要拆掉串行推理保护。
5. 做性能优化时优先查 PaddleOCR 设备选择和回退链路。
6. 提交前至少跑后端 unittest；涉及前端再跑 `npm.cmd run build`。
7. 推送到 GitHub 时使用 `origin/master`，本机 Git push 依赖全局代理 `http://127.0.0.1:7897`。
