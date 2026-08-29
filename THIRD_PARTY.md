# 第三方代码 / 模型声明

本项目整体按 **GNU GPL v3** 发布（见根目录 `LICENSE`），原因之一是移植了
[zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)（GPL-3.0）的部分推理代码。

## 移植自 manga-image-translator（GPL-3.0）

目录：`backend/app/services/engines/mit/`

| 模块 | 来源 | 说明 |
|---|---|---|
| `dbnet.py` / `dbnet_utils.py` / `craft_utils.py` / `imgproc.py` / `rearrange.py` / `resnet34.py` | `detection/default*.py`、`detection/default_utils/*`、`utils/generic.py`（`det_rearrange_forward`/`square_pad_resize`） | 默认 DBNet+ResNet34 文本行检测；`resnet34.py` 为按 torchvision 命名规则自实现，权重名一致 |
| `ctd.py` + `ctd_utils/` | `detection/ctd.py`、`detection/ctd_utils/*` | ComicTextDetector（YOLOv5-s backbone + UNet 掩膜头 + DBNet 行头） |
| `ocr_48px.py` / `xpos.py` | `ocr/model_48px.py`、`ocr/xpos_relative_position.py` | 48px 识别（ConvNeXt + RoFormer/XPOS + beam search） |
| `quadrilateral.py` / `generic2.py` | `utils/generic.py`（Quadrilateral/sort_pnts 等）、`utils/generic2.py` | 几何工具 |
| `bubble.py` | `utils/bubble.py` | `is_ignore` 气泡/拟声词判别 |

以上文件均做了「仅推理」裁剪（移除训练/权重下载框架/日志），并接入本项目的
`app.config` 与 `mit/paths.py`（模型寻径/按需下载）。原版权与许可声明请见
[上游仓库 LICENSE](https://github.com/zyddnys/manga-image-translator/blob/main/LICENSE)（GPL-3.0）。
`DBNet_resnet34.py` 等亦含 NAVER (CRAFT, MIT License) 与 DAWG 的版权声明，按原样保留。

## 模型权重

推理权重不随仓库提交（`backend/data/` 已 gitignore），首次运行按需从
GitHub Releases（`zyddnys/manga-image-translator` `beta-0.3`）自动下载并校验 sha256，
或手动放入 `backend/data/models/mit/`（也可用 `MANGA_MIT_MODEL_DIR` 指向本机 MIT 仓库 `models/`）。

| 权重 | 约大小 | 用途 |
|---|---|---|
| `detection/detect-20241225.ckpt` | 294 MB | default 检测 |
| `detection/comictextdetector.pt` / `.pt.onnx` | ~90-200 MB | ctd 检测（GPU/CPU） |
| `ocr/ocr_ar_48px.ckpt` + `alphabet-all-v7.txt` | 195 MB | 48px 识别 |
| `~/.paddlex/official_models/`（PaddleOCR） | — | PaddleOCR 检测/识别 |