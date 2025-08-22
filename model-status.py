#!/usr/bin/env python3
"""
Model availability probe for HF Inference Providers.

- Reads HF_TOKEN from ~/.env
- Prints basic model info via HfApi.model_info()
- Attempts a minimal chat completion (max_tokens=1)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient


def main():
    parser = argparse.ArgumentParser(description="Probe a Hugging Face model.")
    parser.add_argument(
        "model",
        nargs="?",
        default="tiiuae/Falcon-H1-34B-Instruct",
        help="Model ID to probe (default: tiiuae/Falcon-H1-34B-Instruct).",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help='Provider routing (e.g., "auto", "groq", "together").',
    )
    args = parser.parse_args()

    # Load token
    load_dotenv(os.path.expanduser("~/.env"))
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN not set in ~/.env or environment", file=sys.stderr)
        sys.exit(2)

    # Static info (no deprecation)
    api = HfApi(token=token)
    try:
        info = api.model_info(args.model)
        print(f"Model: {args.model}")
        print(f"private={info.private}, gated={info.gated}, pipeline_tag={info.pipeline_tag}")
        print(f"tags={list(info.tags or [])[:8]}{' ...' if info.tags and len(info.tags) > 8 else ''}")
    except Exception as exc:
        print(f"warning: model_info fetch failed: {exc}", file=sys.stderr)

    # Live 1-token probe
    try:
        client = InferenceClient(api_key=token, provider=args.provider)
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        ok = bool(resp.choices and resp.choices[0].message and resp.choices[0].message.content)
        print(f"probe: {'OK' if ok else 'NO_OUTPUT'}")
    except Exception as exc:
        print(f"probe: FAILED ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()