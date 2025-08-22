#!/usr/bin/env python3
"""
Multi-model CLI probe for concise technical documentation output.

Usage:
  python promptfoo.py -p "Describe the /v1/ingest endpoint..."
  HF_TOKEN must be set in the environment.
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

DEFAULT_MODELS: List[str] = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "01-ai/Yi-1.5-34B-Chat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query multiple HF models with one prompt."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Prompt string to send to each model. Required.",
    )
    parser.add_argument(
        "-m",
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of Hub model IDs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Max tokens in the completion.",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help='Provider routing (e.g., "auto", "groq", "together").',
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="llm_probe_output.txt",
        help="Write results to file. Optional filename, defaults to llm_probe_output.txt.",
    )
    return parser.parse_args()


def make_client(provider: str) -> InferenceClient:
    # Load ~/.env into environment
    load_dotenv(os.path.expanduser("~/.env"))

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("error: HF_TOKEN not set in ~/.env or environment", file=sys.stderr)
        sys.exit(2)
    return InferenceClient(api_key=api_key, provider=provider)


def ask_one(
    client: InferenceClient,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, str]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content.strip()
        latency = f"{time.time() - t0:.3f}s"
        return {"model": model, "latency": latency, "output": content}
    except Exception as exc:  # keep simple for CLI use
        latency = f"{time.time() - t0:.3f}s"
        return {"model": model, "latency": latency, "error": str(exc)}


def main() -> None:
    args = parse_args()

    if not args.prompt:
        print("error: --prompt is required", file=sys.stderr)
        sys.exit(2)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = make_client(provider=args.provider)

    work = []
    with cf.ThreadPoolExecutor(max_workers=len(models)) as ex:
        for m in models:
            work.append(
                ex.submit(
                    ask_one,
                    client,
                    m,
                    args.prompt,
                    args.max_tokens,
                    args.temperature,
                )
            )
        results = [w.result() for w in work]

    # Build output once
    lines = []
    for r in results:
        lines.append("=" * 72)
        lines.append(f"MODEL: {r['model']}")
        lines.append(f"LATENCY: {r['latency']}")
        lines.append("-" * 72)
        if "output" in r:
            lines.append(r["output"])
        else:
            lines.append(f"error: {r['error']}")
    lines.append("=" * 72)
    rendered = "\n".join(lines)

    # Print to terminal
    print(rendered)

    # Optional file write
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(rendered + "\n")
        except OSError as exc:
            print(f"error: failed to write output file: {exc}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
