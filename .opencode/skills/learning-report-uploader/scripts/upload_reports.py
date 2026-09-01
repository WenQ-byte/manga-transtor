#!/usr/bin/env python3
"""
Learning Report Uploader
========================
Scans student situation and daily log directories for Markdown files,
uploads new/changed ones to the learning reports API.
Tracks uploaded files by filename + content hash to avoid duplicates.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ======================== Configuration ========================

API_URL = "http://116.198.217.186:8080/api/v1/agent/learning-reports"

# Skill directory structure
SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
DATA_DIR = SKILL_DIR / "data"
CONFIG_FILE = DATA_DIR / "student_config.json"
UPLOADED_FILE = DATA_DIR / "uploaded_files.json"

# Directories to scan (relative to workspace)
SCAN_DIRS = ["实训记录/学生情况", "实训记录/日志"]

# ======================== Utility Functions ========================

def load_json(path: Path) -> dict:
    """Load JSON from file, return empty dict if not exists."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_json(path: Path, data: dict) -> None:
    """Save JSON to file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file content."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_relative_path(filepath: str, workspace: str) -> str:
    """Get path relative to workspace, using forward slashes."""
    try:
        rel = Path(filepath).relative_to(workspace)
        return str(rel).replace("\\", "/")
    except ValueError:
        return Path(filepath).name


# ======================== Config Management ========================

def get_serial_number() -> str | None:
    """Get student serial number from config file."""
    config = load_json(CONFIG_FILE)
    return config.get("serial_number")


def save_serial_number(serial: str) -> None:
    """Save student serial number to config file."""
    save_json(CONFIG_FILE, {"serial_number": serial})


# ======================== Upload Tracking ========================

def get_uploaded_records() -> list:
    """Get list of uploaded file records."""
    data = load_json(UPLOADED_FILE)
    return data.get("uploaded_files", [])


def find_uploaded_record(filename: str, records: list) -> dict | None:
    """Find an uploaded record by filename."""
    for record in records:
        if record.get("filename") == filename:
            return record
    return None


def add_uploaded_record(filename: str, relative_path: str, content_hash: str,
                         response: dict) -> None:
    """Add or update an uploaded file record."""
    data = load_json(UPLOADED_FILE)
    if "uploaded_files" not in data:
        data["uploaded_files"] = []

    # Remove existing record with same filename (update case)
    data["uploaded_files"] = [
        r for r in data["uploaded_files"] if r.get("filename") != filename
    ]

    data["uploaded_files"].append({
        "filename": filename,
        "relative_path": relative_path,
        "content_hash": content_hash,
        "upload_time": datetime.now().isoformat(),
        "api_response": {
            "success": response.get("success", False),
            "savedCount": response.get("data", {}).get("savedCount", 0),
            "failedCount": response.get("data", {}).get("failedCount", 0),
            "message": response.get("message", "")
        }
    })
    save_json(UPLOADED_FILE, data)


# ======================== File Scanning ========================

def scan_markdown_files(workspace: str) -> list[dict]:
    """
    Scan configured directories for Markdown files.
    Returns list of {filepath, filename, relative_path, content_hash}.
    """
    files = []
    for scan_dir in SCAN_DIRS:
        full_dir = Path(workspace) / scan_dir
        if not full_dir.exists():
            continue
        for md_file in sorted(full_dir.glob("*.md")):
            filepath = str(md_file)
            filename = md_file.name
            relative_path = get_relative_path(filepath, workspace)
            content_hash = compute_file_hash(filepath)
            files.append({
                "filepath": filepath,
                "filename": filename,
                "relative_path": relative_path,
                "content_hash": content_hash
            })
    return files


# ======================== API Upload ========================

def upload_file(serial_number: str, filepath: str) -> dict:
    """
    Upload a single Markdown file to the API using curl.
    Returns the parsed JSON response.
    """
    cmd = [
        "curl", "-s", "-X", "POST", API_URL,
        "-F", f"serial_number={serial_number}",
        "-F", f"files=@{filepath}",
        "--connect-timeout", "15",
        "--max-time", "60"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            return {
                "success": False,
                "message": f"curl error: {result.stderr or 'Unknown error'}"
            }
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "success": False,
                "message": f"Invalid response: {result.stdout[:200]}"
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Request timed out"}
    except Exception as e:
        return {"success": False, "message": f"Exception: {str(e)}"}


# ======================== Main Logic ========================

def main():
    # Get workspace path
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    workspace = os.path.abspath(workspace)

    print(f"=== 学习报告上传 ===")
    print(f"工作目录: {workspace}")
    print()

    # Step 1: Check serial number
    serial = get_serial_number()
    if not serial:
        print("ERROR: 学生序列号未配置")
        print("NEED_SERIAL")
        sys.exit(2)  # Exit code 2 = need serial number

    print(f"学生序列号: {serial}")
    print()

    # Step 2: Scan for Markdown files
    md_files = scan_markdown_files(workspace)
    print(f"扫描到 {len(md_files)} 个 Markdown 文件:")
    for f in md_files:
        print(f"  - {f['relative_path']}")
    print()

    if not md_files:
        print("没有找到可上传的 Markdown 文件")
        print("RESULT: no_files")
        sys.exit(0)

    # Step 3: Check which files need uploading
    uploaded_records = get_uploaded_records()
    files_to_upload = []
    skipped_files = []

    for f in md_files:
        existing = find_uploaded_record(f["filename"], uploaded_records)
        if existing and existing.get("content_hash") == f["content_hash"]:
            skipped_files.append(f)
        else:
            files_to_upload.append(f)

    print(f"需上传: {len(files_to_upload)} 个")
    print(f"已上传(跳过): {len(skipped_files)} 个")
    print()

    if not files_to_upload:
        print("所有文件已上传，无新文件需要上传")
        print("RESULT: all_uploaded")
        sys.exit(0)

    # Step 4: Upload files
    success_count = 0
    fail_count = 0
    failed_files = []

    for f in files_to_upload:
        filename = f["filename"]
        filepath = f["filepath"]
        print(f"正在上传: {filename} ...")

        response = upload_file(serial, filepath)

        if response.get("success"):
            success_count += 1
            add_uploaded_record(filename, f["relative_path"],
                                f["content_hash"], response)
            saved = response.get("data", {}).get("savedCount", 0)
            print(f"  [OK] 成功 (保存 {saved} 个文件)")
        else:
            fail_count += 1
            failed_files.append({"filename": filename, "reason": response.get("message", "Unknown")})
            print(f"  [FAIL] 失败: {response.get('message', 'Unknown error')}")

    # Step 5: Summary
    print()
    print(f"=== 上传完成 ===")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {len(skipped_files)}")

    if failed_files:
        print()
        print("失败详情:")
        for ff in failed_files:
            print(f"  - {ff['filename']}: {ff['reason']}")

    if fail_count > 0:
        print("RESULT: partial_failure")
        sys.exit(1)
    else:
        print("RESULT: success")
        sys.exit(0)


if __name__ == "__main__":
    main()
