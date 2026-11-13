"""Loading the two datasets and deciding whether a response resolves a request.

AmbigNQ (Min et al., EMNLP 2020): open-domain questions, about half ambiguous.
PACIFIC (Deng et al., EMNLP 2022): questions over financial reports (a table plus
text), built from TAT-QA. Some questions were made ambiguous by removing one
entity; the turn that follows supplies it. We use those pairs as ambiguous
requests and ordinary answerable turns as clear requests, each as a standalone
question (PACIFIC's `original_question`) with its report as context.

Every item has: id, dataset, question, ambiguous, intent, answers, and for
PACIFIC also context, scale and answer_type.
"""
import json
import random
import re
import string
from pathlib import Path

SCALES = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 1.0}


# ---------------------------------------------------------------- AmbigNQ

def _annotations(record):
    ann = record["annotations"]
    if not isinstance(ann, dict):          # original JSON release
        return ann
    out = []                               # Hugging Face columnar layout
    for i, kind in enumerate(ann["type"]):
        if kind == "singleAnswer":
            out.append({"type": kind, "answer": list(ann["answer"][i])})
        else:
            qp = ann["qaPairs"][i]
            out.append({"type": kind, "qaPairs": [{"question": q, "answer": list(a)}
                                                  for q, a in zip(qp["question"], qp["answer"])]})
    return out


def _ambignq_item(record, seed):
    first = _annotations(record)[0]
    base = {"id": str(record["id"]), "dataset": "ambignq", "question": record["question"]}
    if first["type"] == "singleAnswer":
        return {**base, "ambiguous": False, "intent": record["question"], "answers": first["answer"]}
    pair = random.Random(f"{seed}:{record['id']}").choice(first["qaPairs"])
    return {**base, "ambiguous": True, "intent": pair["question"], "answers": pair["answer"]}


def load_ambignq(spec):
    if spec.get("local_json"):
        records = json.loads(Path(spec["local_json"]).read_text(encoding="utf-8"))
    else:
        from datasets import load_dataset
        records = list(load_dataset("sewon/ambig_qa", "light", split=spec.get("split", "train")))
    records = sorted(records, key=lambda r: str(r["id"]))
    random.Random(spec["seed"]).shuffle(records)
    return [_ambignq_item(r, spec["seed"]) for r in records[: spec["n_items"]]]


# ---------------------------------------------------------------- PACIFIC

def pacific_context(doc):
    rows = "\n".join(" | ".join(cell for cell in row) for row in doc["table"]["table"])
    text = "\n".join(p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"]))
    return f"Table:\n{rows}\n\nText:\n{text}"


def _answer_list(answer):
    """PACIFIC stores an answer as a list of spans or as a single value."""
    values = answer if isinstance(answer, list) else [answer]
    return [str(v) for v in values]


def pacific_pools(folder, splits):
    """Return (ambiguous items, clear items) from the given PACIFIC splits."""
    ambiguous, clear = [], []
    for split in splits:
        for doc in json.loads((Path(folder) / f"{split}.json").read_text(encoding="utf-8")):
            context = pacific_context(doc)
            turns = sorted(doc["questions"], key=lambda q: q["order"])
            replies = set()
            for i, q in enumerate(turns):
                if q.get("req_clari"):
                    nxt = turns[i + 1] if i + 1 < len(turns) else None
                    if nxt and not nxt.get("req_clari") and nxt.get("answer") \
                            and nxt.get("answer_type") not in (None, "question"):
                        replies.add(nxt["uid"])
                        ambiguous.append({
                            "id": q["uid"], "dataset": "pacific", "ambiguous": True,
                            "question": q["original_question"], "intent": nxt["original_question"],
                            "user_reply": nxt["question"], "answers": _answer_list(nxt["answer"]),
                            "scale": nxt.get("scale", ""), "answer_type": nxt["answer_type"],
                            "context": context, "split": split})
            for q in turns:
                if q["uid"] in replies or q.get("req_clari") or not q.get("answer") \
                        or q.get("answer_type") in (None, "question"):
                    continue
                clear.append({
                    "id": q["uid"], "dataset": "pacific", "ambiguous": False,
                    "question": q["original_question"], "intent": q["original_question"],
                    "answers": _answer_list(q["answer"]), "scale": q.get("scale", ""),
                    "answer_type": q["answer_type"], "context": context, "split": split})
    return ambiguous, clear


def load_pacific(spec):
    ambiguous, clear = pacific_pools(spec["path"], spec.get("splits", ["train", "validation"]))
    rng = random.Random(spec["seed"])
    ambiguous.sort(key=lambda r: r["id"]); clear.sort(key=lambda r: r["id"])
    rng.shuffle(ambiguous); rng.shuffle(clear)
    n_amb = round(spec["n_items"] * spec["ambiguous_share"])
    items = ambiguous[:n_amb] + clear[: spec["n_items"] - n_amb]
    rng.shuffle(items)
    return items


LOADERS = {"ambignq": load_ambignq, "pacific": load_pacific}


def load_items(spec):
    return LOADERS[spec["source"]](spec)


# ---------------------------------------------------------------- scoring

def normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains(response, answer):
    a = normalize(answer)
    return bool(a) and f" {a} " in f" {normalize(response)} "


NUMBER = re.compile(r"\(?-?\$?\d[\d,]*\.?\d*\)?")


WORD_NUMBERS = {w: float(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def numbers_in(text):
    out = [WORD_NUMBERS[w] for w in re.findall(r"[a-z]+", str(text).lower()) if w in WORD_NUMBERS]
    for m in NUMBER.findall(str(text)):
        neg = m.startswith("(") and m.endswith(")") or "-" in m
        digits = m.strip("()$-").replace(",", "").replace("$", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        out.append(-value if neg else value)
    return out


def as_number(answer):
    vals = numbers_in(answer)
    cleaned = re.sub(r"[\s,$%()\-]", "", str(answer))
    return vals[0] if len(vals) == 1 and re.fullmatch(r"\d*\.?\d+", cleaned) else None


def _decimals(x):
    text = f"{x:.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def close(pred, gold):
    """Within 0.5%, or equal once rounded to the precision of either number."""
    if abs(pred - gold) <= max(0.005 * abs(gold), 1e-9):
        return True
    if round(gold, _decimals(pred)) == pred:
        return True
    d = _decimals(gold)
    return d > 0 and round(pred, d) == gold


BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def final_answer(response, split_equals=True):
    """The part of a response that states its final answer.

    Some models show their working despite being told not to. Scoring every number
    in such a reply would give long replies more chances to match, so only the
    final answer is scored: the last bold span if there is one, otherwise the last
    paragraph, and within it the text after the last equals sign.
    """
    text = str(response).strip()
    bold = BOLD.findall(text)
    if bold:
        return bold[-1].strip()
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    last = paragraphs[-1] if paragraphs else text
    if split_equals and "=" in last:
        return last.rsplit("=", 1)[-1].strip()
    return last.strip()


def numeric_match(response, gold, scale):
    mult = SCALES.get(scale or "", 1.0)
    percent = "%" in str(response) or "percent" in str(response).lower()
    for value in numbers_in(response):
        # a ratio can be written as a percentage (47.7% for 0.477) and vice versa
        preds = {value, value / 100} if percent else {value}
        if scale == "percent" and abs(value) <= 1:
            preds.add(value * 100)
        for pred, target in ((p, t) for p in preds for t in {gold, gold * mult}):
            # sign is often expressed in words ("a decrease of 12.5"), so accept the magnitude
            if close(abs(pred), abs(target)):
                return True
    return False


def is_resolved(response, item):
    """True when the response gives the user's intended answer."""
    answers = item["answers"]
    if item.get("dataset") != "pacific":
        return any(contains(response, a) for a in answers)
    span_text = final_answer(response, split_equals=False)   # spans may themselves contain "="
    response = final_answer(response)
    if item.get("answer_type") == "multi-span":
        parts = [a for a in answers if normalize(a)]   # table placeholders such as "-" carry no content
        return bool(parts) and all(
            contains(span_text, a) or (as_number(a) is not None and numeric_match(span_text, as_number(a), item.get("scale")))
            for a in parts)
    for a in answers:
        g = as_number(a)
        if g is not None and numeric_match(response, g, item.get("scale")):
            return True
        if g is None and contains(span_text, a):
            return True
    return False
