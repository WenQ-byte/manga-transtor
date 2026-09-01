# PPTX 生成指南 (增强版)

> 将 HTML 幻灯片转换为可编辑的 .pptx 文件
>
> 参考: joker-duzhong/html-to-pptx 设计思路

## 核心原理

**joker-duzhong/html-to-pptx** 使用浏览器 DOM 的 `getBoundingClientRect()` 获取每个元素的真实渲染位置，将 HTML 元素一一映射为 PPT 形状。

**本 Python 版本** 由于无法访问浏览器 DOM，采用 **CSS 布局类名推断 + 布局规则计算** 来近似定位：

```
CSS 类名（.span-N / .grid-X-Y-Z）→ 布局模式识别 → 位置计算算法 → PPT 坐标
```

---

## 快速开始

### 安装依赖

```bash
pip install python-pptx
```

### 基本用法

```bash
python scripts/html_to_pptx.py input.html output.pptx
```

### 指定风格和主题

```bash
# 风格 A · 电子杂志风 + 墨水经典主题
python scripts/html_to_pptx.py input.html output.pptx A 墨水经典

# 风格 B · 瑞士国际主义风 + IKB蓝
python scripts/html_to_pptx.py input.html output.pptx B IKB蓝
```

---

## 增强特性

### ✅ CSS Grid 布局解析

自动识别 CSS 类名中的 grid 布局模式：

```html
<!-- grid-2-7-5 表示左右两栏 + 右宽栏 -->
<div class="grid-2-7-5">
  <div class="span-2">左侧</div>
  <div class="span-7">中间</div>
  <div class="span-5">右侧</div>
</div>
```

支持的布局模式：

| 类名 | 列比例 | 典型用途 |
|------|--------|----------|
| `grid-2-7-5` | 2:7:5 | 三栏信息页 |
| `grid-2-6-6` | 2:6:6 | 对等双栏 |
| `grid-2-8-4` | 2:8:4 | 主次双栏 |
| `grid-3-3` | 3:3 | 双列对比 |
| `grid-6` | 6 | 六列网格 |

### ✅ Span 类名宽度映射

```html
<!-- span-N 表示占 N/12 的宽度 -->
<div class="span-3">1/4 宽度</div>
<div class="span-4">1/3 宽度</div>
<div class="span-6">1/2 宽度</div>
```

### ✅ 表格转换

自动识别 `<table>` 标签并转换为 PPTX 原生表格：

```html
<table>
  <tr><th>列1</th><th>列2</th></tr>
  <tr><td>数据1</td><td>数据2</td></tr>
</table>
```

### ✅ 列表支持

自动识别 `<ul>` 和 `<ol>` 列表：

```html
<ul>
  <li>第一点</li>
  <li>第二点</li>
</ul>
```

### ✅ 多级标题

支持 `<h1>` 到 `<h6>` 多级标题，自动识别为主标题和副标题。

### ✅ 隐藏元素

使用 `data-html2pptx-ignore` 属性排除不需要的元素：

```html
<div data-html2pptx-ignore>这段内容不会出现在 PPTX 中</div>
```

### ✅ 主题自动识别

根据 HTML class 自动识别主题：

| HTML class | 主题 | 背景色 |
|------------|------|--------|
| `slide light` | 浅色 | 白色/米色 |
| `slide dark` | 深色 | 深灰/黑色 |
| `slide hero light` | 英雄浅色 | 浅色背景 + 大标题 |
| `slide hero dark` | 英雄深色 | 深色背景 + 大标题 |

---

## 风格 A · 主题色

| 主题 | 适合场景 |
|------|----------|
| 墨水经典 | 通用 / 商业发布 / 默认 |
| 靛蓝瓷 | 科技 / 研究 / 数据 / 技术发布会 |
| 森林墨 | 自然 / 可持续 / 文化 / 非虚构 |
| 牛皮纸 | 怀旧 / 人文 / 文学 / 独立杂志 |
| 沙丘 | 艺术 / 设计 / 创意 / 画廊 |

## 风格 B · 锚点色

| 主题 | 颜色 |
|------|------|
| IKB蓝 | 克莱因蓝 #0028FF |
| 柠檬黄 | 柠檬黄 #FFED00 |
| 柠檬绿 | 柠檬绿 #84C100 |
| 安全橙 | 安全橙 #FF6B00 |

---

## 代码调用

```python
from html_to_pptx import html_to_pptx

# 基本调用
html_to_pptx("input.html", "output.pptx")

# 指定风格和主题
html_to_pptx("input.html", "output.pptx", style="A", theme_name="靛蓝瓷")
html_to_pptx("input.html", "output.pptx", style="B", theme_name="IKB蓝")

# 直接传 HTML 字符串
html_to_pptx(html_string, "output.pptx")
```

---

## 转换对应关系

| HTML 结构 | PPTX 输出 |
|-----------|-----------|
| `<section class="slide hero dark">` | 深色背景 + 大标题居中 |
| `<section class="slide hero light">` | 浅色背景 + 大标题居中 |
| `<section class="slide dark">` | 深色背景 + 左对齐内容 |
| `<section class="slide light">` | 浅色背景 + 左对齐内容 |
| `<h1>`-`<h6>` | 标题文字（自动识别主副标题） |
| `<ul>/<ol>` 列表项 | • / — 要点列表 |
| `<table>` | 原生 PPTX 表格 |
| `.grid-X-Y-Z` | 多列布局（位置按列比例计算） |
| `.span-N` | 按 N/12 宽度定位 |
| `data-html2pptx-ignore` | 跳过不转换 |
| `<div class="chrome">` | 自动过滤（页眉元数据） |
| `<div class="foot">` | 自动过滤（页脚） |

---

## 设计原则

### 风格 A · 电子杂志风

- **衬线感标题** → 黑体大号加粗
- **层级分明** → 标题 28pt / 正文 18pt
- **克制配色** → 主题色贯穿始终

### 风格 B · 瑞士国际主义风

- **无衬线极简** → Arial 72pt 主标题
- **单一 accent** → 顶部色条 + 标题强调
- **极致对比** → 大字 72pt vs 正文 20pt

---

## 技术方案对比

### 方案一：Python 脚本

**原理**：CSS 类名推断 + 布局规则计算

```bash
pip install python-pptx
python scripts/html_to_pptx.py input.html output.pptx A 靛蓝瓷
```

| 指标 | 说明 |
|------|------|
| 定位精度 | ★★★☆☆ CSS 类名推断 |
| 依赖 | python-pptx（轻量） |
| 安装 | `pip install python-pptx` |
| 适合场景 | 批量处理、无浏览器环境 |

### 方案二：浏览器方案（推荐高精度）

**原理**：Playwright + Chromium，调用 `getBoundingClientRect()` 获取真实渲染位置

```bash
cd scripts/puppeteer
npm install
npx playwright install chromium
node html_to_pptx_browser.js input.html output.pptx ikb
```

| 指标 | 说明 |
|------|------|
| 定位精度 | ★★★★★ 浏览器真实坐标 |
| 依赖 | Node.js + Playwright + Chromium |
| 安装 | 首次需下载 Chromium (~92MB) |
| 适合场景 | 高还原度需求、单次精确转换 |

**2026-05 修复说明（浏览器方案）**

- 文本提取不再只依赖 `p/h1-h6` 白名单，已扩展为识别瑞士风常见的叶子文本容器，如 `.t-cat`、`.t-meta`、`.ttl`、`.desc`、`.layer-ttl`、`.layer-desc`、`.num-mega` 等。
- 背景色以浏览器 `getComputedStyle()` 的实际结果为准，不再只靠 `hero/dark` 类名猜主题。
- PPT 渲染顺序调整为**先背景 shape，后文本/图片**，同时跳过 `.canvas-card` 这类整页底板容器，避免文字被盖住而看起来像“空白页”。
- 自动忽略 `canvas.ascii-bg`、`i[data-lucide]`、`.chrome-min`、`.foot` 等装饰或元信息节点，减少重复元素和噪音。
- 当前仍不保证 `SVG` 内文本、伪元素、图标路径动画都能转换成可编辑对象；这类内容通常需要在 PowerPoint 中手动补一层。

---

## 设计原则

### 风格 A · 电子杂志风

- **衬线感标题** → 黑体大号加粗
- **层级分明** → 标题 28pt / 正文 18pt
- **克制配色** → 主题色贯穿始终

### 风格 B · 瑞士国际主义风

- **无衬线极简** → Arial 72pt 主标题
- **单一 accent** → 顶部色条 + 标题强调
- **极致对比** → 大字 72pt vs 正文 20pt

---

## 限制说明

### Python 方案限制
- 位置精度依赖 CSS 类名，无法获取浏览器真实渲染坐标
- 复杂 CSS（伪元素、clip-path、filter）无法完美还原
- 依赖 inline style，外部 CSS 需要已内联

### 浏览器方案限制
- 需要安装 Chromium（约 92MB）
- 首次启动较慢（浏览器初始化）

---

## 工作流建议

```
1. 用 guizang-ppt-skill 生成 HTML 网页版 PPT
2. 预览确认结构和内容无误
3. 选择转换方案：
   - 快速批量处理 → Python 脚本
   - 高还原度需求 → 浏览器方案
4. 在 PowerPoint/Keynote/WPS 中微调版式细节
5. 导出最终版本
```

---

## 技术对比

| 方案 | 定位方式 | 精度 | 适用场景 |
|------|----------|------|----------|
| **joker-duzhong/html-to-pptx** | 浏览器 getBoundingClientRect | ★★★★★ | 交互式网页、精确还原 |
| **Python 脚本（本方案）** | CSS 类名推断 | ★★★☆☆ | 自动化流程、AI 集成 |
| **手动复制粘贴** | 人工排版 | ★★☆☆☆ | 少量页面、精确控制 |
