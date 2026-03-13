"""Download a HuggingFace model and patch tokenizer if needed.

Usage:
    python hf-download.py <url_or_repo_id> [--output <dir>] [--patch-tokenizer <class>]

Examples:
    python hf-download.py https://huggingface.co/Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled
    python hf-download.py Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled --output ./my-model
    python hf-download.py Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled --patch-tokenizer Qwen2Tokenizer
"""

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_repo_id(url_or_id: str) -> str:
    """Extract repo ID from a HuggingFace URL or pass through if already a repo ID."""
    match = re.match(r"https?://huggingface\.co/([^/]+/[^/]+?)(?:/.*)?$", url_or_id)
    if match:
        return match.group(1)
    if "/" in url_or_id and not url_or_id.startswith(("http://", "https://")):
        return url_or_id
    print(f"Error: '{url_or_id}' doesn't look like a HuggingFace URL or repo ID.")
    sys.exit(1)


def patch_tokenizer(model_dir: Path, tokenizer_class: str):
    """Patch tokenizer_config.json with a compatible tokenizer class."""
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        print("Warning: No tokenizer_config.json found, skipping patch.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    old_class = config.get("tokenizer_class", "unknown")
    config["tokenizer_class"] = tokenizer_class

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Patched tokenizer_class: {old_class} -> {tokenizer_class}")


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model.")
    parser.add_argument("model", help="HuggingFace URL or repo ID (e.g. user/model)")
    parser.add_argument("--output", "-o", help="Local directory to save model (default: ./<model-name>)")
    parser.add_argument("--patch-tokenizer", "-p", metavar="CLASS",
                        help="Patch tokenizer_class in tokenizer_config.json (e.g. Qwen2Tokenizer)")
    args = parser.parse_args()

    repo_id = parse_repo_id(args.model)
    model_name = repo_id.split("/")[-1]
    output_dir = Path(args.output) if args.output else Path(f"./{model_name}")

    print(f"Downloading {repo_id} -> {output_dir.resolve()}")
    snapshot_download(repo_id=repo_id, local_dir=str(output_dir))
    print("Download complete.")

    if args.patch_tokenizer:
        patch_tokenizer(output_dir, args.patch_tokenizer)

    print(f"\nModel saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
