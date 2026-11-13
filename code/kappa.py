"""Cost per token for each self-hosted tier (kappa, Sections 3.2 and 3.7).

Usage, after collection and benchmark.py for every self-hosted tier of a dataset:
    python kappa.py --dataset pacific

Hourly cost of each self-hosted tier:
  rented   hardware.gpu_hourly_usd, the instance's on-demand price.
  owned    marginal running cost on existing hardware: power x electricity price
           + other_hourly_usd. The hardware is assumed to be in place already, so
           its purchase price is sunk and does not enter a per-request decision.
           Power is the GPU's draw measured by benchmark.py, unless
           owned_hardware.power_kw is set, which overrides it.

Two costs per token are computed for each tier, in US dollars per million tokens:
  run        hourly cost x hours that tier's collection took / tokens it processed.
             The server was lightly used, so this is the cost at low utilization.
             For the owned tier, power is taken from the lowest-load benchmark level.
  full_load  hourly cost / highest throughput in that tier's benchmark, with the
             owned tier's power taken at that same level.

Following Section 3.2, one price per token covers input and output alike.
The script writes, in runs/<dataset>/, kappa.json and config_kappa_run.yaml and
config_kappa_full_load.yaml: copies of the configuration with these prices
filled in. config.yaml is not changed.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from common import SELF_HOSTED, data_dir, load_config, read_jsonl


def number(value, name):
    if not isinstance(value, (int, float)) or value < 0:
        raise SystemExit(f"set {name} in config.yaml to a number")
    return float(value)


def hourly_cost(cfg, t, level):
    """Hourly cost of a tier; `level` is the benchmark level whose power applies (owned tier)."""
    if t["kind"] == "rented":
        value = number(cfg["hardware"].get("gpu_hourly_usd"), "hardware.gpu_hourly_usd")
        return value, {"gpu_hourly_usd": value}
    own = cfg["owned_hardware"]
    price = number(own.get("electricity_usd_per_kwh"), "owned_hardware.electricity_usd_per_kwh")
    other = number(own.get("other_hourly_usd", 0), "owned_hardware.other_hourly_usd")
    if isinstance(own.get("power_kw"), (int, float)):
        power_kw, source = float(own["power_kw"]), "config owned_hardware.power_kw"
    elif level.get("gpu_power_watts_mean"):
        power_kw, source = level["gpu_power_watts_mean"] / 1000, f"measured GPU draw at concurrency {level['concurrency']}"
    else:
        raise SystemExit("no GPU power measured in the benchmark; set owned_hardware.power_kw in config.yaml")
    return power_kw * price + other, {"power_kw": round(power_kw, 4), "power_source": source,
                                      "electricity_per_hour": round(power_kw * price, 6),
                                      "other_per_hour": other}


def calls_for(cfg, key, folder):
    """Every call a self-hosted tier served during collection."""
    calls = []
    if cfg["clarifier"] == key:
        calls += [r["clarify"] for r in read_jsonl(folder / "clarify.jsonl")]
    for r in read_jsonl(folder / f"answers_{key}.jsonl"):
        calls += [r["direct"], r["clarified"]]
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    folder = data_dir(args.dataset, args.data)
    report, prices = {}, {"run": {}, "full_load": {}}
    for t in (t for t in cfg["tiers"] if t["kind"] in SELF_HOSTED):
        calls = calls_for(cfg, t["key"], folder)
        stamps = [datetime.fromisoformat(c["utc"]) for c in calls if c.get("utc")]
        if len(stamps) < 2:
            raise SystemExit(f"no timestamped calls for tier {t['key']}; collect it first")
        hours = (max(stamps) - min(stamps)).total_seconds() / 3600
        if hours <= 0:
            raise SystemExit(f"tier {t['key']}: collection took no measurable time; check the timestamps")
        tokens = sum(c["tin"] + c["tout"] for c in calls)
        bench_path = folder / f"benchmark_{t['key']}.json"
        if not bench_path.exists():
            raise SystemExit(f"missing {bench_path}; run benchmark.py --tier {t['key']}")
        levels = json.loads(bench_path.read_text(encoding="utf-8"))["levels"]
        best = max(levels, key=lambda lv: lv["tokens_per_second"])
        light = min(levels, key=lambda lv: lv["concurrency"])
        hourly_run, parts_run = hourly_cost(cfg, t, light)
        hourly_full, parts_full = hourly_cost(cfg, t, best)
        prices["run"][t["key"]] = hourly_run * hours / tokens * 1e6
        prices["full_load"][t["key"]] = hourly_full / (best["tokens_per_second"] * 3600) * 1e6
        report[t["key"]] = {
            "kind": t["kind"], "model": t["name"],
            "cost_basis": "rental price" if t["kind"] == "rented" else "marginal energy on existing hardware",
            "hourly_cost_usd": {"run": round(hourly_run, 6), "full_load": round(hourly_full, 6)},
            "hourly_cost_parts": {"run": parts_run, "full_load": parts_full},
            "collection_start_utc": min(stamps).isoformat(), "collection_end_utc": max(stamps).isoformat(),
            "collection_hours": round(hours, 4), "tokens_processed": tokens,
            "run_tokens_per_second": round(tokens / (hours * 3600), 2),
            "full_load_concurrency": best["concurrency"],
            "full_load_tokens_per_second": best["tokens_per_second"],
            "gpu_power_watts": {"light_load": light.get("gpu_power_watts_mean"),
                                "full_load": best.get("gpu_power_watts_mean")},
            "kappa_usd_per_million_tokens": {"run": round(prices["run"][t["key"]], 6),
                                             "full_load": round(prices["full_load"][t["key"]], 6)},
        }
    report["note"] = ("run-time kappa covers each tier's collection window only, not setup or idle time; "
                      "the owned tier excludes hardware purchase (sunk) and host power beyond the GPU")
    (folder / "kappa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for basis, values in prices.items():
        variant = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        for t in variant["tiers"]:
            if t["key"] in values:
                t["price_in"] = t["price_out"] = round(values[t["key"]], 6)
                t["price_basis"] = f"kappa_{basis}"
        (folder / f"config_kappa_{basis}.yaml").write_text(yaml.safe_dump(variant, sort_keys=False),
                                                           encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote kappa.json, config_kappa_run.yaml and config_kappa_full_load.yaml in {folder}/")


if __name__ == "__main__":
    main()
