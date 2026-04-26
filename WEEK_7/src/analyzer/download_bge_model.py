"""
将 Hugging Face 上的 BGE 权重完整下载到本地目录，供离线推理或按 FlagEmbedding 官方流程微调。

推理：聚类脚本 --model 可直接指向该目录。
微调：请安装 pip install -U "FlagEmbedding[finetune]"，并参考
https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="snapshot_download BGE 模型到本地目录（用于离线或微调数据准备）",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="Hugging Face repo id，如 BAAI/bge-base-zh-v1.5",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="输出目录（将包含 config、权重等完整快照）",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("需要 huggingface_hub：pip install huggingface_hub") from e

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        max_workers=max(1, args.max_workers),
    )
    meta = {"model": args.model, "local_dir": str(args.local_dir), "snapshot_download_path": path}
    (args.local_dir / "week5_download_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
