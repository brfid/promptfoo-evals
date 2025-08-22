#!/usr/bin/env python3
"""
Multi-model CLI probe for concise technical documentation output.

Usage:
  python llm_probe.py -p "Describe the /v1/ingest endpoint..."
  HF_TOKEN must be set in the environment.
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time
from typing import Dict, List

from huggingface_hub import InferenceClient


DEFAULT_MODELS: List[str] = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "01-ai/Yi-1.5-34B-Chat",
]

DEFAULT_PROMPT = (
    "Write a concise software-technical spec for a REST endpoint "
    "`POST /v1/ingest` that accepts JSON with fields: "
    "`source_id: string`, `events: array<object>`. Include sections for: "
    "Purpose, Request schema, Response schema, Authentication, Error codes, "
    "and one curl example. Keep it under 180 words. Use tight language."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query multiple HF models with one prompt."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt string to send to each model.",
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
        "--timeout",
        type=float,
        default=90.0,
        help="Per-call timeout in seconds.",
    )
    return parser.parse_args()


def make_client(provider: str) -> InferenceClient:
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("error: HF_TOKEN not set in environment", file=sys.stderr)
        sys.exit(2)
    return InferenceClient(api_key=api_key, provider=provider)


def ask_one(
    client: InferenceClient,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, str]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        content = resp.choices[0].message.content.strip()
        latency = f"{time.time() - t0:.3f}s"
        return {"model": model, "latency": latency, "output": content}
    except Exception as exc:  # keep simple for CLI use
        latency = f"{time.time() - t0:.3f}s"
        return {"model": model, "latency": latency, "error": str(exc)}


def main() -> None:
    args = parse_args()
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
                    args.timeout,
                )
            )
        results = [w.result() for w in work]

    # Plain terminal output
    for r in results:
        print("=" * 80)
        print(f"MODEL: {r['model']}")
        print(f"LATENCY: {r['latency']}")
        if "output" in r:
            print("-" * 80)
            print(r["output"])
        else:
            print("-" * 80)
            print(f"error: {r['error']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
