"""一键启动器：选择启动后端 / 前端 / 前后端

用法：
  .venv\\Scripts\\python.exe start.py        # 或双击 start.bat
依赖项目 .venv 与 frontend/node_modules，缺失时会提示。

后端 host/port 与前端 URL 从环境变量 MANGA_HOST / MANGA_PORT
（或项目根目录 .env，或 config.py 默认值 0.0.0.0:8000）读取，不硬编码。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE / ".venv" / "Scripts" / "python.exe"
VENV_OK = PY.exists()
FRONTEND_OK = (BASE / "frontend" / "node_modules").exists()


def _dotenv_values() -> dict[str, str]:
    """读取项目根目录 .env 的 键=值（简单解析，用于 host/port 等内置项）"""
    env_file = BASE / ".env"
    values: dict[str, str] = {}
    if not env_file.is_file():
        return values
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def _get_host_port() -> tuple[str, int]:
    """优先级：进程环境变量 > 项目 .env > 默认值（与 backend/app/config.py 一致）"""
    dotenv = _dotenv_values()
    host = os.environ.get("MANGA_HOST") or dotenv.get("MANGA_HOST") or "0.0.0.0"
    raw_port = os.environ.get("MANGA_PORT") or dotenv.get("MANGA_PORT") or "8000"
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000
    return host, port


HOST, PORT = _get_host_port()
BACKEND_ARGS = [
    str(PY),
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    HOST,
    "--port",
    str(PORT),
    "--app-dir",
    "backend",
]

FRONTEND_ARGS = ["npm.cmd", "run", "dev"]


def _backend_url() -> str:
    host = "localhost" if HOST in ("0.0.0.0", "127.0.0.1") else HOST
    return f"{host}:{PORT}"


def _backend_port_in_use() -> bool:
    """启动前检查端口，避免新 uvicorn 绑定失败后用户继续连到旧进程。"""
    probe_host = "127.0.0.1" if HOST in ("0.0.0.0", "127.0.0.1") else HOST
    try:
        with socket.create_connection((probe_host, PORT), timeout=0.3):
            return True
    except OSError:
        return False


def start_backend() -> int:
    if _backend_port_in_use():
        print(f"[错误] 后端端口 {PORT} 已被占用。请先停止旧后端进程，再重新启动 start.py。\n")
        return 1
    print(f"启动后端 ... API 文档: http://{_backend_url()}/docs  (Ctrl+C 停止)\n")
    return subprocess.call(BACKEND_ARGS, cwd=str(BASE))


def start_frontend() -> int:
    print("启动前端 ... Vite 开发模式: http://localhost:5173  (Ctrl+C 停止)\n")
    return subprocess.call(FRONTEND_ARGS, cwd=str(BASE / "frontend"))


def start_both() -> int:
    if _backend_port_in_use():
        print(f"[错误] 后端端口 {PORT} 已被占用。请先停止旧后端进程，再重新启动 start.py。\n")
        return 1
    print("同时启动后端 + 前端（两个独立终端窗口）...\n")
    print(f"后端: http://{_backend_url()}    前端: http://localhost:5173")
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
    if not (BASE / "backend").is_dir():
        print("[错误] 未找到 backend 目录（启动 uvicorn 需要 --app-dir backend）")
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
        print(f"  [1] 启动后端 (http://{_backend_url()})")
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
