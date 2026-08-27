@echo off
rem 漫画多语言智能翻译系统 - 本地启动脚本
chcp 65001 >nul
echo ==========================================
echo   漫译 · 漫画多语言智能翻译系统
echo ==========================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 创建 Python 虚拟环境...
    python -m venv .venv
)

echo [2/5] 安装后端依赖...
".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt -q

echo [3/5] 安装真实 OCR 依赖（可选，失败可忽略）...
".venv\Scripts\python.exe" -m pip install -r backend\requirements-ai.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>nul

if not exist "frontend\dist\index.html" (
    echo [4/5] 构建前端界面...
    pushd frontend
    call npm install
    call npm run build
    popd
) else (
    echo [4/5] 前端已构建，跳过
)

echo [5/5] 启动服务：http://127.0.0.1:8000
echo        API 文档：http://127.0.0.1:8000/docs
echo        按 Ctrl+C 停止
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend
