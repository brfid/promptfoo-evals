#!/usr/bin/env python3
"""
Multi-model CLI query tool for Hugging Face Inference Providers.

This script lets you send the same prompt to multiple large language models
hosted on Hugging Face Inference Providers and compare their outputs.

Workflow:
    1. Parse command-line arguments including prompt, model list, and options.
    2. Load the Hugging Face API token from ~/.env (HF_TOKEN).
    3. Create an InferenceClient connected to the chosen provider.
    4. Dispatch the prompt concurrently to all selected models.
    5. Collect responses, measure latency, and print results to the terminal.
    6. Optionally save results to a file if the --output flag is provided.

Environment:
    Requires HF_TOKEN to be defined in ~/.env or environment variables.

Usage example:
    python promptfoo.py -p "Summarize the purpose of OAuth2 in 2 sentences."
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

DEFAULT_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Qwen/Qwen3-30B-A3B-Instruct-2507deepseek-ai/DeepSeek-V3.1",
    "Mistral-Nemo-Instruct-2407",
    "google/gemma-3-27b-it",
    #   "CodeGemma-7b-it",  # for coding and code writing
]


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including prompt, models, temperature,
            max_tokens, provider, and optional output filename.
    """
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
        "-t",
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


def make_client(provider):
    """Create an InferenceClient for Hugging Face Inference Providers.

    Args:
        provider (str): Provider routing option (e.g., "auto", "groq", "together").

    Returns:
        InferenceClient: Initialized client for sending inference requests.

    Raises:
        SystemExit: If HF_TOKEN is not set in environment or ~/.env file.
    """
    load_dotenv(os.path.expanduser("~/.env"))

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("error: HF_TOKEN not set in ~/.env or environment", file=sys.stderr)
        sys.exit(2)
    return InferenceClient(api_key=api_key, provider=provider)


def ask_one(client, model, prompt, max_tokens, temperature):
    """Send one prompt to a model and collect its response.

    Args:
        client (InferenceClient): Hugging Face inference client.
        model (str): Model identifier on Hugging Face Hub.
        prompt (str): Prompt string to send.
        max_tokens (int): Maximum tokens in the completion.
        temperature (float): Sampling temperature.

    Returns:
        dict: Keys:
            - model (str): Model ID.
            - latency (str): Time taken in seconds.
            - output (str): Model response content OR detailed error text on failure.
    """
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
    except Exception as exc:
        latency = f"{time.time() - t0:.3f}s"
        # Put failure info where output normally appears
        cls = exc.__class__.__name__
        msg = str(exc) or "no message from provider"
        hint = (
            "Possible causes: model not served by current provider, "
            "cold start disabled, rate limit, or insufficient credits."
        )
        error_text = f"[ERROR] {cls}: {msg}\n{hint}"
        return {"model": model, "latency": latency, "output": error_text}


def main():
    """Main entry point.

    Parses arguments, sends the prompt to multiple models concurrently,
    prints results to the terminal, and optionally writes them to a file.

    Raises:
        SystemExit: If no prompt is provided or output file cannot be written.
    """
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
        lines.append(r["output"])
        lines.append("")
    lines.append("=" * 72)
    rendered = "\n".join(lines)

    print(rendered)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(rendered + "\n")
        except OSError as exc:
            print(f"error: failed to write output file: {exc}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
