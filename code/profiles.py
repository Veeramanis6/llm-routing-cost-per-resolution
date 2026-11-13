"""Derive the cost of a failed attempt (delta) and of a clarifying question (phi)
for each caller profile, from the run's own data and published rates.

    python profiles.py --dataset pacific --config runs/pacific/config_kappa_full_load.yaml --write

Staff profiles value a person's time at a published wage and time each event
from word counts in the data:
    phi   = wage x (read the clarifying question + type a short reply)
    delta = wage x (read the failed answer + retype the request)
Reading 238 words per minute (Brysbaert, 2019, non-fiction, silent reading);
typing 51.56 words per minute (Dhakal et al., CHI 2018). Thinking time and
waiting for the model are left out, so both values are lower bounds.

Real clarification replies are short: the user turns that answer PACIFIC's
clarifying questions average about three words. The reply is therefore timed at
the length of the detail the user adds (PACIFIC's own reply turn; for AmbigNQ,
the words in the intended reading that are not in the question), not at the
length of the full rewritten question that the answering models were shown.

Agent profiles price the agent's own model turn at the frontier tier's posted
prices, with tokens estimated as characters / 4:
    phi           = input (question + clarifying question) + output (reply)
    delta_noticed = input (question + failed answer) + output (reformulated request)
    delta_unnoticed = time for a staff member to find and correct the error later
                      (minutes set below, at the staff wage): a stated scenario.

With --write, the profiles are stored under economics.profiles in the config,
economics.delta and economics.phi are set to the main profile, and the sweep
grids are widened to cover every profile.
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

import yaml

READ_WPM = 238.0         # Brysbaert (2019), J. Memory and Language 109, 104047
TYPE_WPM = 51.56         # Dhakal, Feit, Kristensson, Oulasvirta (2018), CHI '18, paper 646
WAGES = {                # BLS OEWS May 2025 national estimates, released 15 May 2026, median hourly
    "business_financial_operations": 39.74,
    "customer_service_representatives": 21.53,
}
MAIN_WAGE = "business_financial_operations"
LOW_WAGE = "customer_service_representatives"
CORRECTION_MINUTES = 5.0  # scenario: an unnoticed error is found and fixed later by staff
MAIN_PROFILE = "staff"
DELTA_GRID = [0, 0.0002, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
PHI_GRID = [0, 0.0003, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5]


def words(text):
    return len(str(text).split())


def tokens(text):
    return len(str(text)) / 4.0


def added_detail(item):
    """Words the user has to supply: PACIFIC's real reply turn, else the new words of the reading."""
    if item.get("user_reply"):
        return words(item["user_reply"])
    asked = set(re.findall(r"\w+", item["question"].lower()))
    return len([w for w in re.findall(r"\w+", item["intent"].lower()) if w not in asked])


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def derive(folder, cfg):
    items = read_jsonl(folder / "clarify.jsonl")
    responses = [r["direct"]["text"] for t in cfg["tiers"]
                 for r in read_jsonl(folder / f"answers_{t['key']}.jsonl")]
    amb = [i for i in items if i.get("ambiguous")] or items
    q_words = st.mean(words(i["question"]) for i in items)
    cq_words = st.mean(words(i["clarify"]["text"]) for i in items)
    reply_words = st.mean(added_detail(i) for i in amb)
    resp_words = st.mean(words(r) for r in responses)
    t_phi = 60 * (cq_words / READ_WPM + reply_words / TYPE_WPM)
    t_delta = 60 * (resp_words / READ_WPM + q_words / TYPE_WPM)

    api = next(t for t in cfg["tiers"] if t["kind"] == "api")
    p_in, p_out = float(api["price_in"]) / 1e6, float(api["price_out"]) / 1e6
    q_tok = st.mean(tokens(i["question"]) for i in items)
    cq_tok = st.mean(tokens(i["clarify"]["text"]) for i in items)
    reply_tok = st.mean(tokens(i.get("user_reply") or i["intent"]) for i in amb)
    resp_tok = st.mean(tokens(r) for r in responses)
    phi_agent = p_in * (q_tok + cq_tok) + p_out * reply_tok
    delta_agent = p_in * (q_tok + resp_tok) + p_out * q_tok

    per_s = {k: v / 3600 for k, v in WAGES.items()}
    r6 = lambda x: round(x, 6)
    profiles = {
        "staff": {"label": "Staff (business and financial operations wage)",
                  "delta": r6(per_s[MAIN_WAGE] * t_delta), "phi": r6(per_s[MAIN_WAGE] * t_phi)},
        "staff_low_wage": {"label": "Staff (customer service wage)",
                           "delta": r6(per_s[LOW_WAGE] * t_delta), "phi": r6(per_s[LOW_WAGE] * t_phi)},
        "agent_noticed": {"label": "Agent, failure noticed and retried",
                          "delta": r6(delta_agent), "phi": r6(phi_agent)},
        "agent_unnoticed": {"label": "Agent, failure found later by staff",
                            "delta": r6(WAGES[MAIN_WAGE] / 60 * CORRECTION_MINUTES), "phi": r6(phi_agent)},
    }
    basis = {"seconds_to_answer_clarification": round(t_phi, 2), "seconds_to_retry": round(t_delta, 2),
             "words": {"question": round(q_words, 2), "clarifying_question": round(cq_words, 2),
                       "reply_detail": round(reply_words, 2), "response": round(resp_words, 2)},
             "reading_wpm": READ_WPM, "typing_wpm": TYPE_WPM, "wages_usd_per_hour": WAGES,
             "correction_minutes": CORRECTION_MINUTES, "frontier_prices_per_mtok": [api["price_in"], api["price_out"]]}
    return profiles, basis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    parser.add_argument("--config", nargs="+", required=True, help="config file(s) to read, and update with --write")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    folder = Path(args.data) if args.data else Path("runs") / args.dataset
    cfg = yaml.safe_load(open(args.config[0], encoding="utf-8"))
    profiles, basis = derive(folder, cfg)
    print(json.dumps({"profiles": profiles, "basis": basis}, indent=2))
    (folder / "profiles.json").write_text(json.dumps({"profiles": profiles, "basis": basis}, indent=2),
                                           encoding="utf-8")
    if args.write:
        for path in args.config:
            c = yaml.safe_load(open(path, encoding="utf-8"))
            eco = c["economics"]
            eco["profiles"] = profiles
            eco["main_profile"] = MAIN_PROFILE
            eco["delta"], eco["phi"] = profiles[MAIN_PROFILE]["delta"], profiles[MAIN_PROFILE]["phi"]
            eco["delta_grid"], eco["phi_grid"] = DELTA_GRID, PHI_GRID
            Path(path).write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
            print(f"updated {path}")


if __name__ == "__main__":
    main()
