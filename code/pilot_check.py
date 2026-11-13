"""Summarize a pilot (or any partial run) so it can be checked by eye before the full run.

    python pilot_check.py --dataset pacific
    python pilot_check.py --dataset ambignq --samples 5

For the clarify file and each answers_<tier>.jsonl in runs/<dataset>/ it prints
counts, success rates, mean token counts, the model the server reported, a few
sample responses beside the accepted answers, and a projection of the full-run
cost of the frontier tier from the pilot's own token counts.
"""
import argparse
import random
from collections import Counter
from statistics import mean

from common import data_dir, dataset_spec, load_config, read_jsonl


def short(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def tokens(rows, condition, field):
    values = [r[condition].get(field) or 0 for r in rows]
    return mean(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = dataset_spec(cfg, args.dataset)
    folder = data_dir(args.dataset, args.data)
    rng = random.Random(0)

    items = read_jsonl(folder / "clarify.jsonl")
    if not items:
        raise SystemExit(f"no clarify.jsonl in {folder}")
    by_id = {it["id"]: it for it in items}
    n_amb = sum(1 for it in items if it.get("ambiguous"))
    print(f"=== {args.dataset}: {len(items)} items, {n_amb} ambiguous")
    print(f"clarifying questions: mean output tokens {mean(it['clarify'].get('tout') or 0 for it in items):.1f}; "
          f"served by {dict(Counter(it['clarify'].get('served_model') for it in items))}")
    fb = sum(1 for it in items if it["clarify"].get("fallback"))
    fixed = sum(1 for it in items if (it["clarify"].get("attempts") or 1) > 1 and not it["clarify"].get("fallback"))
    print(f"clarifier needed a correction for {fixed} and fell back to the fixed question for {fb} "
          f"of {len(items)} items")
    for it in rng.sample(items, min(args.samples, len(items))):
        print(f"  Q: {short(it['question'])}")
        print(f"  asked: {short(it['clarify']['text'])}"
              + ("   [fallback; model wrote: " + short(it['clarify'].get('model_text', ''), 80) + "]"
                 if it["clarify"].get("fallback") else ""))

    for t in cfg["tiers"]:
        rows = read_jsonl(folder / f"answers_{t['key']}.jsonl")
        if not rows:
            print(f"\n--- {t['key']}: no answers yet")
            continue
        print(f"\n--- {t['key']} ({t['name']}): {len(rows)} items")
        print(f"served by: {dict(Counter(r['direct'].get('served_model') for r in rows))}")
        for cond in ("direct", "clarified"):
            ok = mean(1.0 if r[cond]["ok"] else 0.0 for r in rows)
            stops = Counter(r[cond].get("stop_reason") for r in rows if r[cond].get("stop_reason"))
            if stops:
                print(f"  {cond:9s} stop reasons {dict(stops)}")
            print(f"  {cond:9s} success {ok:5.1%}   mean tokens in {tokens(rows, cond, 'tin'):7.1f}  "
                  f"out {tokens(rows, cond, 'tout'):6.1f}  thinking {tokens(rows, cond, 'thinking_tokens'):6.1f}  "
                  f"empty replies {sum(1 for r in rows if not r[cond]['text'].strip())}")
        for r in rng.sample(rows, min(args.samples, len(rows))):
            it = by_id.get(r["id"], {})
            print(f"  Q: {short(it.get('question', '?'))}")
            print(f"     gold: {short(it.get('answers', '?'), 100)}   scale: {it.get('scale', '')}")
            print(f"     direct    [{'ok' if r['direct']['ok'] else 'x '}] {short(r['direct']['text'], 100)}")
            print(f"     clarified [{'ok' if r['clarified']['ok'] else 'x '}] {short(r['clarified']['text'], 100)}")

        if t.get("kind") == "api" and t.get("price_in") not in (None, "TODO"):
            per_item_in = tokens(rows, "direct", "tin") + tokens(rows, "clarified", "tin")
            per_item_out = tokens(rows, "direct", "tout") + tokens(rows, "clarified", "tout")
            n_full = spec["n_items"]
            cost = n_full * (per_item_in * float(t["price_in"]) + per_item_out * float(t["price_out"])) / 1e6
            print(f"  projected cost of the full {n_full}-item run at this tier: ${cost:,.2f} "
                  f"(output includes any thinking tokens)")


if __name__ == "__main__":
    main()
