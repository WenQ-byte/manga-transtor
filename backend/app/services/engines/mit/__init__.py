"""manga-image-translator 移植模块

以下文件移植自 zyddnys/manga-image-translator（GPL-3.0），已裁剪为仅推理所需：
- detection: default（DBNet+ResNet34）/ ctd（ComicTextDetector）
- ocr: 48px（ConvNeXt + RoFormer 的序列识别模型）
- textline_merge / mask_refinement

项目整体因此为 GPL-3.0，详见根目录 LICENSE。
"""