#!/usr/bin/env python3
"""
SAMI MoE - Benchmark Telemetry Suite
====================================
Measures Time-To-First-Token (TTFT), generation throughput (tok/s), and response latency
against a running local llama-server instance.
Usage:
    python3 benchmark_telemetry.py --runs 3
"""

import sys
import time
import json
import argparse
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"

BENCHMARK_PROMPTS = [
    "Descrie pe scurt principiul de funcționare al unui tranzistor MOSFET.",
    "Explică diferența între paginarea de memorie virtuală și segmentare în sistemele de operare.",
    "Scrie o funcție Python eficientă pentru determinarea numerelor prime până la un N dat."
]

def run_single_benchmark(url: str, prompt: str) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "DeepSeek-MoE-16B-Chat-SAMI-AsymQ",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": True
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    
    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                try:
                    chunk = json.loads(decoded[6:])
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        token_count += 1
                except json.JSONDecodeError:
                    pass

    t_end = time.perf_counter()
    
    ttft = (t_first_token - t_start) if t_first_token else 0.0
    total_time = t_end - t_start
    gen_time = (t_end - t_first_token) if t_first_token else total_time
    tok_per_sec = (token_count / gen_time) if gen_time > 0 else 0.0

    return {
        "tokens": token_count,
        "ttft_sec": ttft,
        "total_time_sec": total_time,
        "tok_per_sec": tok_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="Measure throughput and latency of local SAMI MoE model")
    parser.add_argument("--url", "-u", default=DEFAULT_URL, help="llama-server endpoint URL")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of test runs")
    args = parser.parse_args()

    print("=" * 65)
    print("  🚀 SAMI MoE - BENCHMARK TELEMETRY SUITE")
    print(f"  Target URL: {args.url}")
    print(f"  Total Runs: {args.runs}")
    print("=" * 65)

    results = []
    for i in range(args.runs):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        print(f"\n[*] Run {i+1}/{args.runs}: Sending prompt ({len(prompt)} chars)...")
        try:
            res = run_single_benchmark(args.url, prompt)
            results.append(res)
            print(f"    * Generated: {res['tokens']} tokens")
            print(f"    * TTFT:      {res['ttft_sec']:.2f} s")
            print(f"    * Speed:     {res['tok_per_sec']:.2f} tok/s")
        except Exception as e:
            print(f"    [!] Error during run: {e}")
            sys.exit(1)

    avg_ttft = sum(r["ttft_sec"] for r in results) / len(results)
    avg_speed = sum(r["tok_per_sec"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print("  📊 TELEMETRY SUMMARY")
    print(f"  Average TTFT:       {avg_ttft:.2f} seconds")
    print(f"  Average Throughput: {avg_speed:.2f} tokens/second")
    print("=" * 65)

if __name__ == "__main__":
    main()
