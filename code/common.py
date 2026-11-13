"""Shared helpers: configuration, tiers and loading collected data."""
import json
from pathlib import Path

import yaml

SELF_HOSTED = ("owned", "rented")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = [t["key"] for t in cfg["tiers"]]
    if len(keys) != len(set(keys)):
        raise SystemExit("tier keys in config must be unique")
    if cfg["clarifier"] not in keys:
        raise SystemExit("clarifier must name one of the tiers")
    for tier in cfg["tiers"]:
        if tier["kind"] not in ("owned", "rented", "api"):
            raise SystemExit(f"tier {tier['key']}: kind must be owned, rented or api")
    return cfg


def dataset_spec(cfg, key):
    if key not in cfg.get("datasets", {}):
        raise SystemExit(f"no dataset named {key!r} under datasets: in the config")
    return cfg["datasets"][key]


def data_dir(dataset, override=None):
    """Folder holding one dataset's data files: runs/<dataset>/ unless overridden."""
    return Path(override) if override else Path("runs") / dataset


def tier(cfg, key):
    for t in cfg["tiers"]:
        if t["key"] == key:
            return t
    raise SystemExit(f"no tier named {key!r} in config")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_data(cfg, folder="."):
    """Join clarify.jsonl with answers_<tier>.jsonl for every tier.

    Returns the rows present in every file, each as
    {id, question, ambiguous, intent, answers, clarify, tiers: {key: {direct, clarified}}},
    and a dict of how many rows each tier was missing.
    """
    folder = Path(folder)
    base = {r["id"]: r for r in read_jsonl(folder / "clarify.jsonl")}
    if not base:
        raise SystemExit(f"no clarify.jsonl in {folder.resolve()}")
    answers = {t["key"]: {r["id"]: r for r in read_jsonl(folder / f"answers_{t['key']}.jsonl")}
               for t in cfg["tiers"]}
    missing = {k: len(set(base) - set(v)) for k, v in answers.items()}
    rows = []
    for rid in sorted(base):
        if all(rid in answers[k] for k in answers):
            row = dict(base[rid])
            row["tiers"] = {k: {"direct": answers[k][rid]["direct"],
                                "clarified": answers[k][rid]["clarified"]} for k in answers}
            rows.append(row)
    return rows, missing
