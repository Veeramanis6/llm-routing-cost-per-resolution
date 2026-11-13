"""Measure how many tokens per second a self-hosted tier can serve.

Run on the GPU instance while that tier's model is loaded in the vLLM server:
    python benchmark.py --dataset pacific --tier local
    python benchmark.py --dataset pacific --tier private

Throughput depends on prompt length, so each dataset is benchmarked separately.
It replays that dataset's questions (with their report context, for PACIFIC) at increasing concurrency (requests in flight
at once) and writes benchmark_<tier>.json, including the GPU's average
power draw at each level, read from nvidia-smi. The highest throughput approximates
a fully loaded server; kappa.py turns it into a cost per token.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from collect import WrongModelError, messages_for
from common import SELF_HOSTED, data_dir, dataset_spec, load_config, read_jsonl, tier


def one_call(client, model, item, gen):
    reply = client.chat.completions.create(
        model=model["name"], max_tokens=gen["max_tokens"], temperature=gen["temperature"],
        messages=messages_for(item)[0])
    served = getattr(reply, "model", None)
    if served and served != model["name"]:
        raise WrongModelError(f"server is serving {served!r}, expected {model['name']!r}")
    return reply.usage.prompt_tokens, reply.usage.completion_tokens


class PowerSampler:
    """Reads the GPU's power draw (watts) from nvidia-smi twice a second while running."""

    def __init__(self):
        self.readings, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.readings += [float(x) for x in out.split() if x.replace(".", "", 1).isdigit()]
            except Exception:
                return          # no NVIDIA GPU or tool: leave readings empty
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def mean(self):
        return round(sum(self.readings) / len(self.readings), 1) if self.readings else None


def run_level(client, model, questions, gen, concurrency):
    start = time.perf_counter()
    with PowerSampler() as power, ThreadPoolExecutor(concurrency) as pool:
        usage = list(pool.map(lambda q: one_call(client, model, q, gen), questions))
    seconds = time.perf_counter() - start
    tin = sum(u[0] for u in usage)
    tout = sum(u[1] for u in usage)
    return {"concurrency": concurrency, "requests": len(questions), "seconds": round(seconds, 3),
            "tokens_in": tin, "tokens_out": tout,
            "tokens_per_second": round((tin + tout) / seconds, 2),
            "requests_per_second": round(len(questions) / seconds, 3),
            "gpu_power_watts_mean": power.mean(), "gpu_power_samples": len(power.readings)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    parser.add_argument("--levels", default="1,4,16,64")
    parser.add_argument("--requests", type=int, default=256, help="requests per level")
    args = parser.parse_args()

    from openai import OpenAI
    cfg = load_config(args.config)
    model = tier(cfg, args.tier)
    if model["kind"] not in SELF_HOSTED:
        raise SystemExit("benchmark only applies to self-hosted tiers")
    client = OpenAI(base_url=model["base_url"], api_key=os.environ[model["api_key_env"]])
    folder = data_dir(args.dataset, args.data)
    spec = dataset_spec(cfg, args.dataset)
    gen = {"temperature": cfg["generation"]["temperature"], "max_tokens": spec["max_tokens"]}
    questions = read_jsonl(folder / "clarify.jsonl")
    if not questions:
        raise SystemExit(f"no questions found; run collect.py --dataset {args.dataset} --stage clarify first")

    one_call(client, model, questions[0], gen)   # warm-up, not timed
    levels = []
    for level in (int(x) for x in args.levels.split(",")):
        batch = [questions[i % len(questions)] for i in range(args.requests)]
        result = run_level(client, model, batch, gen, level)
        levels.append(result)
        print(result)

    info = {"utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tier": model["key"], "model": model["name"], "revision": model.get("revision"),
            "dataset": args.dataset, "hardware": cfg.get("hardware", {}), "generation": gen,
            "levels": levels}
    out = folder / f"benchmark_{model['key']}.json"
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
