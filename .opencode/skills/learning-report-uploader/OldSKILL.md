---
name: learning-report-uploader
description: |
  学习报告上传技能 — 自动扫描实训知识库中的学生情况和日志 Markdown 文件，
  通过 API 上传到学习报告系统。支持定时自动上传和手动触发上传。
  触发条件：当用户提到「上传学习报告」「上传学生情况」「上传日志」「同步学习报告」
  「上传报告」等与学习报告上传相关的内容时，或由定时自动化任务触发时。
  核心功能：文件去重（基于内容哈希）、学生序列号管理、批量上传、上传结果报告。
---

# 学习报告上传技能

## 概述

本技能负责将实训知识库中的学生情况和日志 Markdown 文件上传到学习报告系统。
- **API 地址**：`POST http://116.198.217.186:8080/api/v1/agent/learning-reports`
- **扫描目录**：`实训知识库/学生情况/` 和 `实训知识库/日志/`
- **数据文件**：本技能 `data/` 目录下
  - `student_config.json` — 学生序列号配置
  - `uploaded_files.json` — 已上传文件记录（含内容哈希）

## 工作流程

### 步骤 1：检查学生序列号

1. 读取本技能 `data/student_config.json` 文件
2. 如果 `serial_number` 字段为空或不存在：
   - **必须询问用户**："请提供你的学生序列号（serial_number），用于上传学习报告"
   - 用户提供后，保存到 `data/student_config.json`：`{"serial_number": "用户提供的值"}`
   - 如果用户拒绝或无法提供，终止上传流程并告知原因
3. 如果序列号已存在，继续下一步

### 步骤 2：运行上传脚本

使用 Bash 工具执行上传脚本。**所有路径使用工作区相对路径，不硬编码绝对路径**：

```bash
python .workbuddy/skills/learning-report-uploader/scripts/upload_reports.py .
```

- `.workbuddy/skills/learning-report-uploader/scripts/upload_reports.py` — 上传脚本（相对于工作区根目录）
- `.` — 工作区根目录作为参数传入（脚本会在此目录下扫描 `实训知识库/学生情况/` 和 `实训知识库/日志/`）

脚本会自动完成：
- 扫描 `实训知识库/学生情况/` 和 `实训知识库/日志/` 下的 `.md` 文件
- 对比 `data/uploaded_files.json` 中的记录（基于文件名 + 内容哈希）
- 跳过内容未变化的已上传文件
- 上传新增或内容有变化的文件
- 更新 `data/uploaded_files.json` 记录

### 步骤 3：报告上传结果

根据脚本输出，向用户报告：
- 本次扫描到多少个文件
- 新增/变化文件数
- 成功上传数、失败数
- 失败原因（如有）

## 触发方式

### 自动触发（定时任务）

建议设置 3 个每日定时检查点（由自动化任务驱动）：
- **10:00** — 上午检查上传
- **14:00** — 下午检查上传
- **18:00** — 傍晚检查上传

**随机延迟（防止并发峰值）**：当多位同学同时使用定时器时，API 服务器可能瞬间承受大量并发请求。因此每个定时任务在上传前必须加入 0~300 秒（0~5 分钟）的随机延迟，将请求分散到时间窗口中。

定时任务 prompt 示例：
```
步骤 0（随机延迟）：执行 Bash：sleep $((RANDOM % 301))，在 0~300 秒（0~5 分钟）内随机等待。

步骤 1：检查学生序列号：读取 .workbuddy/skills/learning-report-uploader/data/student_config.json。

步骤 2：如果序列号存在，运行 Bash：python .workbuddy/skills/learning-report-uploader/scripts/upload_reports.py .
如果脚本输出 NEED_SERIAL，跳过本次上传。

步骤 3：报告上传结果（成功/失败/跳过数量）。
```

定时任务触发时，直接执行步骤 1 ~ 3，无需用户交互（序列号已配置的情况下）。

### 手动触发

用户说以下内容时触发：
- "上传学习报告" / "上传学生情况" / "上传日志"
- "同步学习报告" / "同步学生数据"
- "帮我上传报告" / "上传一下"

## 异常处理

- **序列号不存在**：API 返回 `"Student not found with serial_number: xxx"`，告知用户序列号可能有误
- **网络错误**：脚本会报告 curl 错误，告知用户稍后重试或检查网络
- **无新文件**：脚本输出 "所有文件已上传，无新文件"，正常结束
- **序列号为空**：必须先获取序列号才能继续

## 数据文件格式

### student_config.json

```json
{
  "serial_number": null
}
```

### uploaded_files.json

```json
{
  "uploaded_files": [
    {
      "filename": "张三.md",
      "relative_path": "实训知识库/学生情况/张三.md",
      "content_hash": "a1b2c3d4...",
      "upload_time": "2026-07-09T10:00:00",
      "api_response": { "success": true, "savedCount": 1 }
    }
  ]
}
```

## 重要约束

1. **序列号是必须的** — 没有序列号绝不上传，必须先询问用户
2. **内容去重** — 文件名相同且内容哈希相同则跳过；内容变化则重新上传
3. **幂等性** — 多次运行不会重复上传相同内容的文件
4. **本地优先** — 序列号和上传记录都保存在本地技能目录，不依赖外部状态
5. **工作区相对路径** — 所有路径引用使用工作区相对路径，确保其他同学可直接使用
