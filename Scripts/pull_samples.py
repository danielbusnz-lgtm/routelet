"""Pull redacted distillation samples from the proxy's R2 bucket into one local
JSONL file.

The aegis proxy stores one immutable JSON object per sample under
``samples/<date>/<device>/<ts>-<uuid>.json`` (see aegis proxy/handlers/routelet.ts).
R2 has no append, so the per-sample-object layout sidesteps write races across
devices; compaction into a single batched JSONL happens here, at pull time.

This is a full snapshot: every object under the prefix is read and written out,
deduped by R2 key. Re-running overwrites the output. The raw text was already
redacted on-device and scrubbed again by the proxy, but it is still real user
input, so the output is gitignored (see data/raw/).

Reads R2 over the S3-compatible API. Set in the environment (or .env):
    R2_ACCOUNT_ID          Cloudflare account id
    R2_ACCESS_KEY_ID       R2 API token access key
    R2_SECRET_ACCESS_KEY   R2 API token secret
    R2_BUCKET              bucket name (default: aegis-routelet-samples)

Create a read-only R2 API token in the Cloudflare dashboard
(R2 > Manage API Tokens). boto3 is a tooling dep installed into the venv, not a
project dependency.

Usage:
    uv run python Scripts/pull_samples.py
    uv run python Scripts/pull_samples.py --prefix samples/2026-05-29/ --out data/raw/today.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "samples.jsonl"
DEFAULT_BUCKET = "aegis-routelet-samples"
DEFAULT_PREFIX = "samples/"


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"missing ${name}. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, and "
            f"R2_SECRET_ACCESS_KEY (read-only R2 API token)."
        )
    return val


def r2_client():
    account_id = _require_env("R2_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=_require_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_require_env("R2_SECRET_ACCESS_KEY"),
        # R2 ignores the region but boto3 requires one; signature v4 is required.
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def pull(bucket: str, prefix: str, out: Path) -> tuple[int, int]:
    """Read every object under prefix and write one JSON line per valid sample.
    Returns (written, skipped)."""
    client = r2_client()
    paginator = client.get_paginator("list_objects_v2")

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out.open("w") as f:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
                try:
                    sample = json.loads(body)
                except json.JSONDecodeError:
                    print(f"  skip (bad json): {key}", file=sys.stderr)
                    skipped += 1
                    continue
                # One object is one sample; the key already dedupes, so write as-is.
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="R2 key prefix to pull")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL path")
    args = parser.parse_args()

    written, skipped = pull(args.bucket, args.prefix, args.out)
    print(f"wrote {written} samples to {args.out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
