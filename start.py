"""一键启动器：选择启动后端 / 前端 / 前后端

用法：
  .venv\\Scripts\\python.exe start.py        # 或双击 start.bat
依赖项目 .venv 与 frontend/node_modules，缺失时会提示。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE / ".venv" / "Scripts" / "python.exe"
VENV_OK = PY.exists()
FRONTEND_OK = (BASE / "frontend" / "node_modules").exists()

BACKEND_ARGS = [
    str(PY),
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--app-dir",
    "backend",
]

FRONTEND_ARGS = ["npm.cmd", "run", "dev"]


def start_backend() -> int:
    print("启动后端 ... API 文档: http://localhost:8000/docs  (Ctrl+C 停止)\n")
    return subprocess.call(BACKEND_ARGS, cwd=str(BASE))


def start_frontend() -> int:
    print("启动前端 ... Vite 开发模式: http://localhost:5173  (Ctrl+C 停止)\n")
    return subprocess.call(FRONTEND_ARGS, cwd=str(BASE / "frontend"))


def start_both() -> int:
    print("同时启动后端 + 前端（两个独立终端窗口）...\n")
    print("后端: http://localhost:8000    前端: http://localhost:5173")
    print("关闭对应窗口即停止服务。\n")
    backend = subprocess.Popen(BACKEND_ARGS, cwd=str(BASE), creationflags=subprocess.CREATE_NEW_CONSOLE)
    frontend = subprocess.Popen(FRONTEND_ARGS, cwd=str(BASE / "frontend"), creationflags=subprocess.CREATE_NEW_CONSOLE)
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()
    return 0


def check_env() -> bool:
    if not VENV_OK:
        print("[错误] 未找到 .venv\\Scripts\\python.exe，请先创建虚拟环境并安装依赖：")
        print("  python -m venv .venv")
        print("  .venv\\Scripts\\pip install -r backend\\requirements.txt")
        print("  .venv\\Scripts\\pip install -r backend\\requirements-ai.txt")
        print("  .venv\\Scripts\\pip install -r backend\\requirements-inpaint.txt")
        return False
    return True


def check_frontend() -> bool:
    if not FRONTEND_OK:
        print("[错误] 未找到 frontend\\node_modules，请先安装前端依赖：")
        print("  cd frontend")
        print("  npm install")
        print("  cd ..")
        return False
    return True


def main() -> None:
    while True:
        print()
        print("==========================================")
        print("  漫译 · 漫画多语言智能翻译系统")
        print("==========================================")
        print("  [1] 启动后端 (端口 8000)")
        print("  [2] 启动前端 (Vite 开发模式)")
        print("  [3] 同时启动前后端")
        print("  [0] 退出")
        print("==========================================")
        try:
            choice = input("请选择 [0-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if choice == "1":
            if check_env():
                start_backend()
        elif choice == "2":
            if check_frontend():
                start_frontend()
        elif choice == "3":
            if check_env() and check_frontend():
                start_both()
        elif choice == "0":
            break
        else:
            print("无效输入，请输入 0-3")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
