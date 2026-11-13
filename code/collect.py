"""Collect model outcomes for the token-economics routing study, one stage at a time.

Stage "clarify" samples the dataset's questions, attaches the simulated user's
intent, and has the clarifier tier write one clarifying question for each.
    python collect.py --dataset pacific --stage clarify

Stage "answer" has one tier answer every question twice: as asked, and after the
clarifying question and the simulated user's reply.
    python collect.py --dataset pacific --stage answer --tier local

Outputs go to runs/<dataset>/ (or the folder given by --data):
    clarify.jsonl           questions, intents, accepted answers, clarifying questions
    answers_<tier>.jsonl    one line per question for that tier
    manifest_collect_<stage>_<tier>_<time>.json   environment of each sitting

Every stage can be stopped and restarted with the same command; finished
questions are skipped. Use --limit for a pilot.

Self-hosted tiers share one GPU, so run their stages one after the other with
the matching model loaded in vLLM; the script refuses to continue if the server
reports a different model than the config names.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common import SELF_HOSTED, data_dir, dataset_spec, load_config, read_jsonl, tier
from datasets_io import is_resolved, load_items
from manifest import write_manifest

ANSWER_SYSTEM = "Answer the question with a short phrase. Do not explain."
ANSWER_SYSTEM_DOC = ("Answer the question using the report below. Reply with only the final answer: "
                     "a number with its unit, or a short phrase. Do not explain or show calculations.")
CLARIFY_SYSTEM = (
    "You speak to a user before their question is answered. Do not answer the question. "
    "Write one short clarifying question that would tell you exactly what the user wants, "
    "for example which year, which item or which meaning they intend. "
    "Output only that question, ending with a question mark.\n\n"
    "Example. User question: What was the change in revenue?\n"
    "Your output: Between which two years do you want the change in revenue?\n\n"
    "Example. User question: Who played the lead in the film?\n"
    "Your output: Which film, and which version of it, do you mean?")
CLARIFY_RETRY = "That was not a question. Reply with one clarifying question only, ending with a question mark."
CLARIFY_FALLBACK = "Could you say exactly which item, period or meaning you have in mind?"
CLARIFY_ATTEMPTS = 3        # first try plus two corrections; then the fallback question


class WrongModelError(RuntimeError):
    pass


# ---------------------------------------------------------------- prompts

def user_text(item, question):
    if item.get("context"):
        return f"{item['context']}\n\nQuestion: {question}"
    return question


def messages_for(item):
    system = ANSWER_SYSTEM_DOC if item.get("context") else ANSWER_SYSTEM
    direct = [{"role": "system", "content": system},
              {"role": "user", "content": user_text(item, item["question"])}]
    clarified = direct + [{"role": "assistant", "content": item["clarify"]["text"]},
                          {"role": "user", "content": f"I mean: {item['intent']}"}]
    return direct, clarified


def clarify_messages(item):
    return [{"role": "system", "content": CLARIFY_SYSTEM},
            {"role": "user", "content": user_text(item, item["question"])}]


# ---------------------------------------------------------------- model calls

def make_client(model):
    key = os.environ[model["api_key_env"]]
    if model.get("provider") == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=key)
    from openai import OpenAI
    return OpenAI(base_url=model["base_url"], api_key=key)


def _call_openai(client, model, messages, max_tokens, gen):
    reply = client.chat.completions.create(model=model["name"], messages=messages,
                                           max_tokens=max_tokens, temperature=gen["temperature"])
    return (reply.choices[0].message.content or "", reply.usage.prompt_tokens,
            reply.usage.completion_tokens, None, getattr(reply, "model", None),
            getattr(reply.choices[0], "finish_reason", None))


def _call_anthropic(client, model, messages, max_tokens, gen):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    kwargs = {"model": model["name"], "max_tokens": max_tokens, "system": system, "messages": turns}
    if model.get("effort"):
        kwargs["output_config"] = {"effort": model["effort"]}
    if model.get("inference_geo"):
        kwargs["inference_geo"] = model["inference_geo"]
    # no temperature: Claude 4.7 and later reject values below 1
    reply = client.messages.create(**kwargs)
    text = "".join(b.text for b in reply.content if getattr(b, "type", "") == "text")
    details = getattr(reply.usage, "output_tokens_details", None)
    thinking = getattr(details, "thinking_tokens", None) if details else None
    return (text, reply.usage.input_tokens, reply.usage.output_tokens, thinking,
            getattr(reply, "model", None), getattr(reply, "stop_reason", None))


def chat(client, model, messages, max_tokens, gen):
    call = _call_anthropic if model.get("provider") == "anthropic" else _call_openai
    for attempt in range(6):
        try:
            text, tin, tout, thinking, served, stop = call(client, model, messages, max_tokens, gen)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
            continue
        if model["kind"] in SELF_HOSTED and served and served != model["name"]:
            raise WrongModelError(f"server is serving {served!r}, config expects {model['name']!r} "
                                  f"for tier {model['key']!r}; restart vLLM with the right model")
        return {"text": text, "tin": tin, "tout": tout, "thinking_tokens": thinking,
                "served_model": served,      # the exact version served; API names can be aliases
                "stop_reason": stop,
                "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "retries": attempt}


def max_tokens_for(model, spec):
    return model.get("max_tokens") or spec["max_tokens"]


def is_question(text):
    """A usable clarifying question: one short sentence ending in a question mark."""
    t = text.strip().strip('"').strip()
    return t.endswith("?") and len(t.split()) <= 60


def clarify_item(item, client, model, spec, gen):
    """Ask the clarifier for a question; correct it up to twice if it answers instead.

    Tokens of every attempt are summed, since the firm pays for all of them. If no
    attempt yields a question, a fixed neutral question is used and flagged, and the
    fallback rate is reported.
    """
    messages = clarify_messages(item)
    tin = tout = 0
    out = None
    for attempt in range(1, CLARIFY_ATTEMPTS + 1):
        out = chat(client, model, messages, spec["max_tokens"], gen)
        tin += out["tin"] or 0
        tout += out["tout"] or 0
        if is_question(out["text"]):
            break
        messages = messages + [{"role": "assistant", "content": out["text"]},
                               {"role": "user", "content": CLARIFY_RETRY}]
    fallback = not is_question(out["text"])
    clar = {**out, "text": CLARIFY_FALLBACK if fallback else out["text"].strip().strip('"').strip(),
            "tin": tin, "tout": tout, "attempts": attempt, "fallback": fallback}
    if fallback:
        clar["model_text"] = out["text"]
    return {**item, "clarify": clar}


def answer_item(item, client, model, spec, gen):
    direct, clarified = messages_for(item)
    row = {"id": item["id"], "tier": model["key"]}
    for condition, messages in (("direct", direct), ("clarified", clarified)):
        out = chat(client, model, messages, max_tokens_for(model, spec), gen)
        out["ok"] = is_resolved(out["text"], item)
        row[condition] = out
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True, help="a key under datasets: in the config")
    parser.add_argument("--stage", choices=["clarify", "answer"], required=True)
    parser.add_argument("--tier", help="tier key, required for --stage answer")
    parser.add_argument("--data", default=None, help="folder for the data files (default runs/<dataset>)")
    parser.add_argument("--limit", type=int, default=None, help="pilot size")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = dataset_spec(cfg, args.dataset)
    key = cfg["clarifier"] if args.stage == "clarify" else args.tier
    if not key:
        raise SystemExit("--tier is required for --stage answer")
    model = tier(cfg, key)
    client = make_client(model)
    gen = cfg["generation"]
    folder = data_dir(args.dataset, args.data)
    folder.mkdir(parents=True, exist_ok=True)

    if args.stage == "clarify":
        out_path = folder / "clarify.jsonl"
        items = load_items(spec)[: args.limit]
        work_fn = lambda it: clarify_item(it, client, model, spec, gen)
    else:
        out_path = folder / f"answers_{key}.jsonl"
        items = read_jsonl(folder / "clarify.jsonl")[: args.limit]
        if not items:
            raise SystemExit(f"run --stage clarify for {args.dataset} first")
        work_fn = lambda it: answer_item(it, client, model, spec, gen)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_manifest(folder / f"manifest_collect_{args.stage}_{key}_{stamp}.json",
                   f"collect {args.dataset} {args.stage} {key}", cfg, args.config)
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.dataset}: stage {args.stage}, tier {key} ({model['name']}): {len(done)} done, {len(todo)} to go")

    lock = threading.Lock()

    def work(item):
        row = work_fn(item)
        with lock, out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        for count, _ in enumerate(pool.map(work, todo), 1):
            if count % 50 == 0:
                print(f"{count}/{len(todo)}")
    print("finished")


if __name__ == "__main__":
    main()
