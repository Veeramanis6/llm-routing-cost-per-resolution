"""Manual audit of the automatic scorer, using the Python standard library only.

Step 1, draw a blind sample (the automatic verdicts are kept in a separate key file):
    python audit.py --dataset ambignq --data runs/ambignq --sample 100

Open audit_<dataset>.csv in Excel or Numbers. For each row, read the question,
the user's intended reading and the accepted answers, then the response, and put
y (the response answers the intended question correctly) or n in the human_ok
column. Save it as CSV with the same name.

Step 2, compare your verdicts with the scorer's:
    python audit.py --dataset ambignq --data runs/ambignq --score

It reports agreement, how often the scorer marked a correct answer wrong (missed
paraphrases) and how often it marked a wrong answer right, overall and by tier.
These rates go in the paper next to the success rates they qualify.

--data can point at an unzipped run bundle (run_<dataset>_<time>/) on any computer.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def draw(folder, dataset, n, seed):
    items = {r["id"]: r for r in read_jsonl(folder / "clarify.jsonl")}
    cells = defaultdict(list)                     # (tier, condition) -> candidate rows
    for path in sorted(folder.glob("answers_*.jsonl")):
        for r in read_jsonl(path):
            for cond in ("direct", "clarified"):
                cells[(r["tier"], cond)].append((r, cond))
    rng = random.Random(seed)
    keys = sorted(cells)
    per_cell = -(-n // len(keys))                 # stratified: about equal rows per tier and condition
    picked = []
    for key in keys:
        pool = cells[key]
        picked += rng.sample(pool, min(per_cell, len(pool)))
    rng.shuffle(picked)
    picked = picked[:n]

    out_csv = folder / f"audit_{dataset}.csv"
    key_file = folder / f"audit_{dataset}_key.json"
    key = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "question", "intended_reading", "accepted_answers", "response", "human_ok"])
        for i, (r, cond) in enumerate(picked, 1):
            it = items[r["id"]]
            reply = it["clarify"]["text"] if cond == "clarified" else ""
            w.writerow([i, it["question"], it.get("intent", ""), " | ".join(map(str, it["answers"])),
                        r[cond]["text"], ""])
            key[str(i)] = {"id": r["id"], "tier": r["tier"], "condition": cond, "auto_ok": bool(r[cond]["ok"]),
                           "clarifying_question": reply}
    key_file.write_text(json.dumps({"seed": seed, "rows": key}, indent=1), encoding="utf-8")
    print(f"wrote {out_csv} ({len(picked)} rows) and {key_file}")


def score(folder, dataset):
    key = json.loads((folder / f"audit_{dataset}_key.json").read_text(encoding="utf-8"))["rows"]
    rows = list(csv.DictReader(open(folder / f"audit_{dataset}.csv", encoding="utf-8-sig")))
    judged = []
    for r in rows:
        h = (r.get("human_ok") or "").strip().lower()
        if h in ("y", "yes", "1", "true"):
            judged.append((key[r["row"]], True))
        elif h in ("n", "no", "0", "false"):
            judged.append((key[r["row"]], False))
    if not judged:
        raise SystemExit("no y/n verdicts found in the human_ok column")

    def summarise(pairs):
        n = len(pairs)
        agree = sum(k["auto_ok"] == h for k, h in pairs)
        human_right = [k for k, h in pairs if h]
        human_wrong = [k for k, h in pairs if not h]
        missed = sum(not k["auto_ok"] for k in human_right)       # correct, scored wrong
        false_ok = sum(k["auto_ok"] for k in human_wrong)          # wrong, scored correct
        return {"n": n, "agreement": agree / n, "agreement_95ci": wilson(agree, n),
                "human_success": len(human_right) / n, "auto_success": sum(k["auto_ok"] for k, _ in pairs) / n,
                "missed_correct": missed, "missed_rate_among_correct": missed / len(human_right) if human_right else None,
                "false_correct": false_ok, "false_rate_among_wrong": false_ok / len(human_wrong) if human_wrong else None}

    report = {"overall": summarise(judged)}
    by_tier = defaultdict(list)
    for k, h in judged:
        by_tier[k["tier"]].append((k, h))
    report["by_tier"] = {t: summarise(p) for t, p in sorted(by_tier.items())}
    out = folder / f"audit_{dataset}_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score", action="store_true", help="compare filled-in verdicts with the scorer")
    args = parser.parse_args()
    folder = Path(args.data) if args.data else Path("runs") / args.dataset
    if args.score:
        score(folder, args.dataset)
    else:
        draw(folder, args.dataset, args.sample, args.seed)


if __name__ == "__main__":
    main()
