"""Train the embedding router and evaluate routing policies on one dataset's outcomes.

Usage (after kappa.py has written the priced configs):
    python analyze.py --dataset pacific
    python analyze.py --dataset pacific --config runs/pacific/config_kappa_run.yaml --out runs/pacific/results_kappa_run

With three tiers there are six actions: answer at each tier, or ask a
clarifying question and then answer at each tier.

Writes CSV tables to the output folder (default runs/<dataset>/results/):
    descriptive.csv                    success rate and mean cost of every action
    estimator_quality.csv              AUC and Brier score of every success estimator
    policies.csv                       every policy, mean and SD over the splits
    policies_by_ambiguity.csv          the same, for ambiguous and unambiguous questions separately
    policies_reweighted.csv            the same, reweighted to the dataset's natural ambiguity rate
    policies_reweighted_prior_corrected.csv  the same, with the router retrained on training requests
                                       weighted to the natural rate (robustness check)
    policies_by_profile.csv, bootstrap_by_profile.csv  every policy under each caller profile
    policies_reweighted*_by_profile.csv, bootstrap_reweighted_by_profile.csv  the same at the natural rate
                                       (only when natural_ambiguous_share is set)
    bootstrap.csv                      paired bootstrap of the full router against key baselines
    sensitivity_<name>.csv             one assumption varied: delta, phi, price_scale (API tier),
                                       utilization (self-hosted tiers), owned_cost_scale
                                       (owned tier: unmeasured host power and overheads)
    manifest_analyze.json              hardware, software versions, models and prices of this run
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from common import data_dir, dataset_spec, load_config, load_data
from manifest import write_manifest

Q_FLOOR = 0.02   # keeps the expected-cost formula finite when a predicted probability is near zero


class Space:
    """The actions available: answer directly at each tier, or ask first and then answer."""

    def __init__(self, cfg):
        self.tiers = [t["key"] for t in cfg["tiers"]]
        self.kind = {t["key"]: t["kind"] for t in cfg["tiers"]}
        self.actions = [(k, "direct") for k in self.tiers] + [(k, "clarified") for k in self.tiers]
        self.names = [f"answer_{k}" if c == "direct" else f"ask_then_{k}" for k, c in self.actions]
        self.asked = np.array([c == "clarified" for _, c in self.actions], dtype=int)
        self.tier_of = np.array([self.tiers.index(k) for k, _ in self.actions])
        self.api = next(k for k in self.tiers if self.kind[k] == "api")
        self.local = next(k for k in self.tiers if self.kind[k] == "owned")
        self.private = next(k for k in self.tiers if self.kind[k] == "rented")
        self.policies = ([f"always_{k}" for k in self.tiers] + [f"always_ask_{k}" for k in self.tiers]
                         + ["random", "router_without_ask", "router_with_ask", "oracle"])

    def index(self, key, condition):
        return self.actions.index((key, condition))

    def tier_mask(self, key):
        return self.tier_of == self.tiers.index(key)


def tier_prices(cfg, api_scale=1.0, utilization=1.0, owned_scale=1.0):
    prices = {}
    for t in cfg["tiers"]:
        try:
            pin, pout = float(t["price_in"]), float(t["price_out"])
        except (TypeError, ValueError):
            raise SystemExit(f"tier {t['key']} has no price; analyse a config written by kappa.py")
        factor = api_scale if t["kind"] == "api" else 1.0 / utilization
        if t["kind"] == "owned":
            factor *= owned_scale
        prices[t["key"]] = (pin * factor, pout * factor)
    return prices


def build(rows, cfg, space, api_scale=1.0, utilization=1.0, owned_scale=1.0):
    """C: attempt cost per action, A: cost of writing the question, Y: resolved (0/1)."""
    prices = tier_prices(cfg, api_scale, utilization, owned_scale)
    usd = lambda key, call: (call["tin"] * prices[key][0] + call["tout"] * prices[key][1]) / 1e6
    n = len(rows)
    C, Y = np.zeros((n, len(space.actions))), np.zeros((n, len(space.actions)), dtype=int)
    A = np.array([usd(cfg["clarifier"], r["clarify"]) for r in rows])
    for i, r in enumerate(rows):
        for j, (key, condition) in enumerate(space.actions):
            call = r["tiers"][key][condition]
            C[i, j], Y[i, j] = usd(key, call), int(call["ok"])
    return C, A, Y


def embed(texts, name, folder):
    key = hashlib.md5(("\n".join(texts) + name).encode("utf-8")).hexdigest()[:12]
    cache = folder / f"embeddings_{key}.npy"
    if cache.exists():
        return np.load(cache)
    if name == "tfidf":   # smoke tests only: fitted on all questions, not for reported results
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        sparse = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
        X = TruncatedSVD(n_components=min(128, sparse.shape[1] - 1), random_state=0).fit_transform(sparse)
    else:
        from sentence_transformers import SentenceTransformer
        X = SentenceTransformer(name).encode(texts, batch_size=64, show_progress_bar=True,
                                             normalize_embeddings=True)
    np.save(cache, X)
    return X


def fit_predict(X_train, Y_train, X_test, sample_weight=None):
    """One logistic regression per action; returns predicted success probabilities.

    sample_weight, when given, reweights training requests (used to correct for a
    sample whose share of ambiguous requests differs from the natural share).
    """
    Q = np.zeros((len(X_test), Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        y = Y_train[:, j]
        if y.min() == y.max():
            Q[:, j] = y[0]
        else:
            Q[:, j] = LogisticRegression(max_iter=2000).fit(X_train, y, sample_weight=sample_weight) \
                .predict_proba(X_test)[:, 1]
    return Q


def expected_cost(Q, mean_attempt_cost, mean_ask_cost, delta, phi, asked):
    """Expected cost per resolved request for each action (Sections 3.3 and 3.5).

    mean_attempt_cost may be one row (all requests) or one row per request.
    """
    q = np.clip(Q, Q_FLOOR, 1.0)
    m = np.atleast_2d(mean_attempt_cost)
    retry = (m + (1 - q) * delta) / q
    return retry + asked[None, :] * (mean_ask_cost + phi)


def choose(policy, space, J, cost, Y, rng, allowed):
    n = len(J)
    masked = lambda M: np.where(allowed, M, np.inf)
    if policy.startswith("always_"):
        ask = policy.startswith("always_ask_")
        key = policy[len("always_ask_"):] if ask else policy[len("always_"):]
        condition = "clarified" if ask else "direct"
        t0 = space.tiers.index(key)
        a = np.full(n, space.index(key, condition))
        # a request this policy may not send to its tier goes to the nearest permitted
        # tier with the same action (cheaper tier first on ties); with every action
        # permitted, as in all analyses here, this never applies
        order = sorted(range(len(space.tiers)), key=lambda t: (abs(t - t0), t))
        for i in np.where(~allowed[np.arange(n), a])[0]:
            for t in order:
                j = space.index(space.tiers[t], condition)
                if allowed[i, j]:
                    a[i] = j
                    break
        return a
    if policy == "random":
        draw = rng.random((n, len(space.actions)))
        draw[:, space.asked == 1] = np.inf
        return masked(draw).argmin(1)
    if policy == "router_without_ask":
        Jd = masked(J).copy()
        Jd[:, space.asked == 1] = np.inf
        return Jd.argmin(1)
    if policy == "router_with_ask":
        return masked(J).argmin(1)
    if policy == "oracle":   # cheapest permitted action that resolves; cheapest permitted if none does
        c = masked(cost)
        hit = np.where(Y == 1, c, np.inf)
        return np.where(np.isfinite(hit).any(1), hit.argmin(1), c.argmin(1))
    raise ValueError(policy)


def score(space, actions, cost, Y, delta, w=None):
    """Weighted metrics; cost already includes the question and its friction for the ask actions."""
    idx = np.arange(len(actions))
    if len(actions) == 0:
        return {}
    w = np.ones(len(actions)) if w is None else w
    c, y = cost[idx, actions], Y[idx, actions]
    total = (w * (c + (1 - y) * delta)).sum()
    resolved = (w * y).sum()
    out = {"cost_per_resolved": total / resolved if resolved else np.inf,
           "resolution_rate": resolved / w.sum(),
           "cost_per_request": (w * c).sum() / w.sum(),
           "share_ask": (w * space.asked[actions]).sum() / w.sum()}
    for t, key in enumerate(space.tiers):
        out[f"share_tier_{key}"] = (w * (space.tier_of[actions] == t)).sum() / w.sum()
    return out


def evaluate(space, fits, C, A, Y, ambiguous, eco, delta=None, phi=None, weights=None):
    """Score every policy on every split, overall and by ambiguity."""
    delta = eco["delta"] if delta is None else delta
    phi = eco["phi"] if phi is None else phi
    records, first = [], None
    for seed, train, test, Q in fits:
        allowed = np.ones((len(test), len(space.actions)), bool)   # every action permitted
        factor = np.ones((len(test), len(space.actions)))
        cost = C[test] * factor + space.asked[None, :] * (A[test][:, None] + phi)
        J = expected_cost(Q, C[train].mean(0)[None, :] * factor, A[train].mean(), delta, phi, space.asked)
        rng = np.random.default_rng(seed)
        acts = {p: choose(p, space, J, cost, Y[test], rng, allowed) for p in space.policies}
        w = None if weights is None else weights[test]
        groups = {"all": np.ones(len(test), bool), "ambiguous": ambiguous[test], "unambiguous": ~ambiguous[test]}
        for p, a in acts.items():
            for g, m in groups.items():
                if m.any():
                    records.append({"seed": seed, "subset": g, "policy": p,
                                    **score(space, a[m], cost[m], Y[test][m], delta, None if w is None else w[m])})
        if first is None:
            first = (test, acts, cost)
    return pd.DataFrame(records), first


def summarize(table, subset="all"):
    t = table[table["subset"] == subset].drop(columns=["seed", "subset"])
    grouped = t.groupby("policy", sort=False)
    out = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd"))
    return out[sorted(out.columns)].reset_index()


def paired_bootstrap(space, first, Y, delta, samples, baselines, weights=None):
    test, acts, cost = first
    y = Y[test]
    w = None if weights is None else weights[test]
    draws = np.random.default_rng(0).integers(0, len(test), (samples, len(test)))
    rows = []
    for base in baselines:
        wd = (lambda d: None) if w is None else (lambda d: w[d])
        diffs = [score(space, acts["router_with_ask"][d], cost[d], y[d], delta, wd(d))["cost_per_resolved"]
                 - score(space, acts[base][d], cost[d], y[d], delta, wd(d))["cost_per_resolved"] for d in draws]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"comparison": f"router_with_ask minus {base}",
                     "mean_difference": float(np.mean(diffs)), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None, help="default runs/<dataset>/config_kappa_full_load.yaml")
    parser.add_argument("--data", default=None, help="default runs/<dataset>")
    parser.add_argument("--out", default=None, help="default runs/<dataset>/results")
    args = parser.parse_args()

    folder = data_dir(args.dataset, args.data)
    config_path = args.config or folder / "config_kappa_full_load.yaml"
    cfg = load_config(config_path)
    spec = dataset_spec(cfg, args.dataset)
    eco, rt = cfg["economics"], cfg["router"]
    space = Space(cfg)
    out = Path(args.out) if args.out else folder / "results"
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest_analyze.json", f"analyze {args.dataset}", cfg, config_path)
    rows, missing = load_data(cfg, folder)
    if any(missing.values()):
        print(f"note: questions missing per tier (excluded): {missing}")
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} complete questions; collect every tier first")
    C, A, Y = build(rows, cfg, space)
    ambiguous = np.array([r["ambiguous"] for r in rows])

    pd.DataFrame({
        "action": space.names,
        "success_rate": Y.mean(0),
        "success_rate_ambiguous": Y[ambiguous].mean(0) if ambiguous.any() else np.nan,
        "success_rate_unambiguous": Y[~ambiguous].mean(0) if (~ambiguous).any() else np.nan,
        "mean_cost_usd": (C + space.asked[None, :] * A[:, None]).mean(0),
    }).to_csv(out / "descriptive.csv", index=False)
    print(f"{args.dataset}: {len(rows)} requests, {ambiguous.mean():.1%} ambiguous, {len(space.actions)} actions")

    # the router sees only the question text, never the report context or the answer
    X = embed([r["question"] for r in rows], rt["embedding_model"], out)
    fits = []
    for seed in rt["seeds"]:
        train, test = train_test_split(np.arange(len(rows)), test_size=rt["test_fraction"], random_state=seed)
        fits.append((seed, train, test, fit_predict(X[train], Y[train], X[test])))

    quality = []
    for seed, _, test, Q in fits:
        for j, name in enumerate(space.names):
            y = Y[test, j]
            quality.append({"seed": seed, "estimator": name,
                            "auc": roc_auc_score(y, Q[:, j]) if y.min() != y.max() else np.nan,
                            "brier": brier_score_loss(y, Q[:, j])})
    pd.DataFrame(quality).groupby("estimator", sort=False).mean().drop(columns="seed") \
        .reset_index().to_csv(out / "estimator_quality.csv", index=False)

    table, first = evaluate(space, fits, C, A, Y, ambiguous, eco)
    summary = summarize(table)
    summary.to_csv(out / "policies.csv", index=False)
    pd.concat([summarize(table, g).assign(subset=g) for g in ("ambiguous", "unambiguous")
               if (table["subset"] == g).any()]).to_csv(out / "policies_by_ambiguity.csv", index=False)
    natural = spec.get("natural_ambiguous_share")
    weights, fits_pc = None, None
    if natural and 0 < ambiguous.mean() < 1:
        weights = np.where(ambiguous, natural / ambiguous.mean(), (1 - natural) / (1 - ambiguous.mean()))
        summarize(evaluate(space, fits, C, A, Y, ambiguous, eco, weights=weights)[0]) \
            .assign(natural_ambiguous_share=natural).to_csv(out / "policies_reweighted.csv", index=False)
        # robustness: the same router retrained with the training requests weighted to the
        # natural share (prior-shift correction), scored on the reweighted test requests
        fits_pc = [(seed, train, test, fit_predict(X[train], Y[train], X[test], weights[train]))
                   for seed, train, test, _ in fits]
        summarize(evaluate(space, fits_pc, C, A, Y, ambiguous, eco, weights=weights)[0]) \
            .assign(natural_ambiguous_share=natural).to_csv(out / "policies_reweighted_prior_corrected.csv", index=False)
    paired_bootstrap(space, first, Y, eco["delta"], rt["bootstrap_samples"],
                     [f"always_{space.api}", f"always_{space.local}", "router_without_ask"]) \
        .to_csv(out / "bootstrap.csv", index=False)

    keep = ["policy", "cost_per_resolved_mean", "resolution_rate_mean", "share_ask_mean"] + \
           [f"share_tier_{k}_mean" for k in space.tiers]

    # caller profiles (profiles.py): each sets its own delta and phi
    profiles = eco.get("profiles") or {}
    if profiles:
        by_profile, boots, reweighted, corrected, boots_nat = [], [], [], [], []
        for name, p in profiles.items():
            t_p, first_p = evaluate(space, fits, C, A, Y, ambiguous, eco, delta=p["delta"], phi=p["phi"])
            info = {"profile": name, "label": p.get("label", name), "delta": p["delta"], "phi": p["phi"]}
            by_profile.append(summarize(t_p).assign(**info))
            boots.append(paired_bootstrap(space, first_p, Y, p["delta"], rt["bootstrap_samples"],
                                          [f"always_{space.api}", f"always_{space.local}", "router_without_ask"])
                         .assign(**info))
            if weights is not None:
                bases = [f"always_{space.api}", "router_without_ask"]
                for label, fset, sink in (("trained on sample", fits, reweighted),
                                          ("prior-corrected", fits_pc, corrected)):
                    t_w, first_w = evaluate(space, fset, C, A, Y, ambiguous, eco, delta=p["delta"],
                                            phi=p["phi"], weights=weights)
                    sink.append(summarize(t_w).assign(**info))
                    boots_nat.append(paired_bootstrap(space, first_w, Y, p["delta"], rt["bootstrap_samples"],
                                                      bases, weights).assign(router=label, **info))
        pd.concat(by_profile).to_csv(out / "policies_by_profile.csv", index=False)
        pd.concat(boots).to_csv(out / "bootstrap_by_profile.csv", index=False)
        if reweighted:
            pd.concat(reweighted).assign(natural_ambiguous_share=natural) \
                .to_csv(out / "policies_reweighted_by_profile.csv", index=False)
            pd.concat(corrected).assign(natural_ambiguous_share=natural) \
                .to_csv(out / "policies_reweighted_prior_corrected_by_profile.csv", index=False)
            pd.concat(boots_nat).assign(natural_ambiguous_share=natural) \
                .to_csv(out / "bootstrap_reweighted_by_profile.csv", index=False)

    def sweep(name, grid, run, extra=None):
        parts = []
        for value in grid:
            part = summarize(run(value))[keep]
            part.insert(0, name, value)
            if extra:
                for k, v in extra.items():
                    part.insert(0, k, v)
            parts.append(part)
        return pd.concat(parts)

    ev = lambda C_, A_, Y_, **kw: evaluate(space, fits, C_, A_, Y_, ambiguous, eco, **kw)[0]
    sweeps = {
        "delta": (eco["delta_grid"], lambda v: ev(C, A, Y, delta=v)),
        "phi": (eco["phi_grid"], lambda v: ev(C, A, Y, phi=v)),
        "price_scale": (eco["price_scale_grid"], lambda v: ev(*build(rows, cfg, space, api_scale=v))),
        "utilization": (eco["utilization_grid"], lambda v: ev(*build(rows, cfg, space, utilization=v))),
        "owned_cost_scale": (eco.get("owned_cost_scale_grid", [1.0]),
                             lambda v: ev(*build(rows, cfg, space, owned_scale=v))),
    }
    for name, (grid, run) in sweeps.items():
        sweep(name, grid, run).to_csv(out / f"sensitivity_{name}.csv", index=False)

    print(summary[keep].to_string(index=False))
    print(f"tables written to {out}/")


if __name__ == "__main__":
    main()
