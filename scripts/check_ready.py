"""Print which submission requirements are already satisfied."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import SYSTEM_NAME, SYSTEM_VERSION, get_settings

settings = get_settings()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def main() -> None:
    commit = git_commit()
    rows = [
        ("系统名称", SYSTEM_NAME, True),
        ("版本号", SYSTEM_VERSION, True),
        ("本地评测密钥 MEMORY_API_KEY", "已写入 .env" if settings.api_key else "空", bool(settings.api_key)),
        ("DashScope text-embedding-v4", "已配置" if settings.dashscope_api_key else "未提供", bool(settings.dashscope_api_key)),
        (
            "gpt-4o-mini 抽取",
            "已配置" if settings.llm_extract and settings.openai_api_key and settings.llm_model == "gpt-4o-mini" else "未提供 OpenAI 密钥",
            bool(settings.llm_extract and settings.openai_api_key and settings.llm_model == "gpt-4o-mini"),
        ),
        ("公开仓库的 git commit", commit or "还没有提交", bool(commit)),
        ("公网 HTTPS 地址", "未部署", False),
    ]
    for name, detail, ok in rows:
        mark = "ready" if ok else "need-you"
        print(f"[{mark}] {name}: {detail}")


if __name__ == "__main__":
    main()
