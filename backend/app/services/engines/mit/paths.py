"""MIT 模型权重寻径/下载"""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import get_settings

# 已知的 GitHub Release 权重（对应 MIT repo beta-0.3）
REMOTE_MODELS = {
    "detection/detect-20241225.ckpt": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt",
        "hash": "67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e",
    },
    "detection/comictextdetector.pt": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt",
        "hash": "1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb",
    },
    "detection/comictextdetector.pt.onnx": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
        "hash": "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
    },
    "ocr/ocr_ar_48px.ckpt": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt",
        "hash": "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d",
    },
    "ocr/alphabet-all-v7.txt": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt",
        "hash": "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a",
    },
}


def _candidates(rel: str) -> list[Path]:
    s = get_settings()
    cands = []
    if s.mit_model_dir:
        cands.append(Path(s.mit_model_dir) / rel)
    if s.mit_fallback_dir:
        cands.append(Path(s.mit_fallback_dir) / rel)
    # 常见本机 MIT 仓库目录（开发回退）
    mit_repo = Path("D:/Develop/code/Python/manga-image-translator/models") / rel
    if mit_repo.exists():
        cands.append(mit_repo)
    return cands


def resolve_model(rel: str) -> Path:
    """按优先级返回已存在的模型路径；全部缺失返回主目录预期路径（供下载/报错）"""
    for p in _candidates(rel):
        if p.is_file():
            return p
    s = get_settings()
    base = Path(s.mit_model_dir) if s.mit_model_dir else Path(s.data_dir) / "models" / "mit"
    return base / rel


def model_ready(rel: str) -> bool:
    return resolve_model(rel).is_file()


def ensure_downloaded(rel: str) -> Path:
    """模型缺失时按需下载到主模型目录，返回路径；下载失败抛异常"""
    path = resolve_model(rel)
    if path.is_file():
        return path
    if rel not in REMOTE_MODELS:
        raise FileNotFoundError(f"模型缺失且无下载地址: {rel} → {path}")
    import requests

    meta = REMOTE_MODELS[rel]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    print(f"下载模型 {rel} ...")
    r = requests.get(meta["url"], stream=True, timeout=300)
    r.raise_for_status()
    with open(temp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    h = hashlib.sha256()
    with open(temp, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    if meta.get("hash") and h.hexdigest() != meta["hash"]:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"模型 {rel} sha256 校验失败")
    temp.replace(path)
    return path