#!/usr/bin/env python3
"""
SAMI MoE - Simple Inference Client Example
==========================================
Demonstrates streaming OpenAI-compatible completions from a local llama-server instance.
Usage:
    python3 client_example.py --prompt "Explică pe scurt conceptul de experți partajați."
"""

import sys
import json
import argparse
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"

def query_model(prompt: str, url: str = DEFAULT_URL, temperature: float = 0.3):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "DeepSeek-MoE-16B-Chat-SAMI-AsymQ",
        "messages": [
            {"role": "system", "content": "You are a concise, technically rigorous assistant. Răspunde direct și structurat."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": True
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    print(f"\n[*] Prompt: {prompt}\n" + "-" * 60)

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    try:
                        chunk = json.loads(decoded[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print("\n" + "-" * 60)
    except urllib.error.URLError as e:
        print(f"\n[!] Connection Error: {e}")
        print("    Ensure llama-server is running on http://127.0.0.1:8080")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Query local SAMI MoE model via llama-server")
    parser.add_argument("--prompt", "-p", default="Explică eficiența unui model Mixture of Experts pe memorie DDR4.", help="Prompt text")
    parser.add_argument("--url", "-u", default=DEFAULT_URL, help="llama-server endpoint URL")
    parser.add_argument("--temp", "-t", type=float, default=0.3, help="Sampling temperature")

    args = parser.parse_args()
    query_model(args.prompt, url=args.url, temperature=args.temp)

if __name__ == "__main__":
    main()
