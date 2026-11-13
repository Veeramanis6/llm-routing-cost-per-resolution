"""Seal a finished run into one archive with checksums and a filled-in run record.

Run this once per dataset, after analyze.py, from the code folder:
    python capture.py --dataset pacific --notes "anything unusual"

It reads runs/<dataset>/ and writes, inside a new folder runs/<dataset>/run_<dataset>_<UTC time>/ :
    RUN_RECORD.md        the facts for the reproducibility table of the paper
    environment.txt      exact versions of every installed Python package
    checksums.sha256     SHA-256 of every file in the bundle
and then zips that folder. Send the zip back, and keep a copy unchanged:
the checksums let anyone confirm later that the data were not altered.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import data_dir, load_config, read_jsonl

CODE_FILES = ["common.py", "datasets_io.py", "collect.py", "analyze.py", "manifest.py", "capture.py", "benchmark.py",
              "kappa.py", "pilot_check.py", "audit.py", "profiles.py", "reprofile.py", "requirements.txt", "make_figures.py", "README.md"]
DATA_PATTERNS = ["config.yaml", "config_kappa_*.yaml", "config_hosted*.yaml", "clarify.jsonl",
                 "answers_*.jsonl", "benchmark_*.json", "kappa.json", "manifest_collect_*.json",
                 "vllm_*.log", "audit_*.csv", "audit_*_key.json", "audit_*_summary.json", "profiles.json"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tier_calls(cfg, folder):
    """All logged calls per tier, the clarifying questions counted under the clarifier."""
    calls = {t["key"]: [] for t in cfg["tiers"]}
    calls[cfg["clarifier"]] += [r["clarify"] for r in read_jsonl(folder / "clarify.jsonl")]
    for key in calls:
        for r in read_jsonl(folder / f"answers_{key}.jsonl"):
            calls[key] += [r["direct"], r["clarified"]]
    return calls


def fmt_price(t):
    return f"{t.get('price_in')} / {t.get('price_out')}"


def fmt_kappa(k):
    if not k:
        return "n/a"
    return f"run {k['run']:.4f}; full load {k['full_load']:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None, help="default runs/<dataset>/config_kappa_full_load.yaml")
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    parser.add_argument("--results", default=None, help="default runs/<dataset>/results")
    parser.add_argument("--notes", default="", help="anything unusual about the run")
    args = parser.parse_args()

    here = data_dir(args.dataset, args.data).resolve()
    args.config = str(args.config or here / "config_kappa_full_load.yaml")
    cfg = load_config(args.config)
    results = Path(args.results) if args.results else here / "results"
    if not (results / "policies.csv").exists():
        sys.exit("results folder has no policies.csv; run analyze.py first")
    questions = read_jsonl(here / "clarify.jsonl")
    if not questions:
        sys.exit("no clarify.jsonl found")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = here / f"run_{args.dataset}_{stamp}"
    (bundle / "code").mkdir(parents=True)
    shutil.copy2(args.config, bundle / Path(args.config).name)
    shutil.copytree(results, bundle / "results", ignore=shutil.ignore_patterns("embeddings_*.npy"))
    for pattern in DATA_PATTERNS:
        for f in sorted(here.glob(pattern)):
            if f.is_file() and not (bundle / f.name).exists():
                shutil.copy2(f, bundle / f.name)
    for extra in sorted(here.glob("results_*")):
        if extra.is_dir():
            shutil.copytree(extra, bundle / extra.name, ignore=shutil.ignore_patterns("embeddings_*.npy"))
    code_dir = Path(__file__).resolve().parent
    if (code_dir / "config.yaml").exists():
        shutil.copy2(code_dir / "config.yaml", bundle / "config.yaml")
    for name in CODE_FILES:
        if (code_dir / name).exists():
            shutil.copy2(code_dir / name, bundle / "code" / name)

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (bundle / "environment.txt").write_text(f"python {sys.version}\n\n{freeze}", encoding="utf-8")
    (bundle / "question_ids.txt").write_text("\n".join(sorted(r["id"] for r in questions)) + "\n",
                                             encoding="utf-8")

    detected = {}
    if (results / "manifest_analyze.json").exists():
        detected = json.loads((results / "manifest_analyze.json").read_text(encoding="utf-8")).get("detected", {})
    gpu_seen = sorted({g for m in here.glob("manifest_collect_*.json")
                       for g in json.loads(m.read_text(encoding="utf-8"))["detected"].get("gpus", [])})
    kappa = json.loads((here / "kappa.json").read_text(encoding="utf-8")) if (here / "kappa.json").exists() else {}
    hw, own = cfg.get("hardware", {}), cfg.get("owned_hardware", {})

    calls = tier_calls(cfg, here)
    thinking_total = sum(c.get("thinking_tokens") or 0 for c in calls.get(next(
        (t["key"] for t in cfg["tiers"] if t["kind"] == "api"), ""), []))
    clar = [q.get("clarify", {}) for q in questions]
    fallbacks = sum(1 for c in clar if c.get("fallback"))
    corrected = sum(1 for c in clar if (c.get("attempts") or 1) > 1 and not c.get("fallback"))
    tier_rows, all_times, retries = [], [], 0
    for t in cfg["tiers"]:
        c = calls[t["key"]]
        served = Counter(x.get("served_model") or "not reported" for x in c)
        times = [x["utc"] for x in c if x.get("utc")]
        all_times += times
        retries += sum(x.get("retries", 0) for x in c)
        k = kappa.get(t["key"], {}).get("kappa_usd_per_million_tokens")
        tier_rows.append(
            f"| {t['key']} ({t['kind']}) | {t['name']} (revision {t.get('revision')}) | "
            f"{'; '.join(f'{m} ({n})' for m, n in served.most_common())} | {len(c):,} | "
            f"{sum(x['tin'] for x in c):,} / {sum(x['tout'] for x in c):,} | {fmt_price(t)} | "
            f"{fmt_kappa(k)} | "
            f"{(min(times) + ' to ' + max(times)) if times else 'n/a'} |")

    todo = [f"hardware.{k}" for k, v in hw.items() if v == "TODO"] + \
           [f"owned_hardware.{k}" for k, v in own.items() if v == "TODO"] + \
           [f"tiers.{t['key']}.{f}" for t in cfg["tiers"] for f in ("name", "revision") if t.get(f) == "TODO"]
    record = f"""# Run record

Created {stamp} by capture.py. Every value below is read from the run's own files.

## Hardware and software (Section 5.7)

| Item | Value |
| --- | --- |
| Compute provider and instance | {hw.get('provider')} / {hw.get('instance')} |
| GPU rental price per hour (USD) | {hw.get('gpu_hourly_usd')} |
| GPU detected during collection | {', '.join(gpu_seen) or 'none detected'} |
| Local tier cost basis | marginal energy on existing hardware; power {own.get('power_kw') or 'measured GPU draw'}; electricity {own.get('electricity_usd_per_kwh')} USD/kWh; other {own.get('other_hourly_usd')} USD/h |
| Analysis machine | {detected.get('os')}; {detected.get('cpu_count')} cores; {detected.get('ram_gb')} GB RAM |
| Python | {detected.get('python')} |
| Sentence encoder | {cfg['router']['embedding_model']} |
| Dataset | {args.dataset} (sampling seed {cfg['datasets'][args.dataset]['seed']}) |
| Seeds | splits {cfg['router']['seeds']} |
| Thinking tokens billed, frontier tier | {thinking_total:,} |
| Clarifying questions: corrected once or twice / replaced by the fallback question | {corrected} / {fallbacks} of {len(clar)} |
| Questions sampled | {len(questions)} |
| Collection window, all tiers (UTC) | {(min(all_times) + ' to ' + max(all_times)) if all_times else 'n/a'} |
| Calls retried after an error | {retries} |
| Config used for the analysis | {Path(args.config).name}, SHA-256 {sha256(args.config)} |

## Tiers

| Tier | Model requested | Model actually served (calls) | Calls | Tokens in / out | Prices in config (USD per M tokens, in / out) | Measured cost per token (USD per M) | Collection window (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(tier_rows)}

## Notes

{args.notes or 'none'}

## Checks before submitting

- Fields still marked TODO: {', '.join(todo) if todo else 'none'}
- More than one served version for a tier means the model changed during the run;
  report both and say so in Section 5.2.
- Exact package versions are in environment.txt.
"""
    (bundle / "RUN_RECORD.md").write_text(record, encoding="utf-8")

    lines = [f"{sha256(p)}  {p.relative_to(bundle).as_posix()}"
             for p in sorted(bundle.rglob("*")) if p.is_file()]
    (bundle / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(bundle), "zip", bundle.parent, bundle.name)
    print(record)
    print(f"bundle: {archive}")
    print(f"archive SHA-256: {sha256(archive)}  (write this down)")


if __name__ == "__main__":
    main()
