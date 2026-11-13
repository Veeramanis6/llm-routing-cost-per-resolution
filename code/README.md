# Code

The pipeline behind the experiments in "Answer, Ask or Escalate: Token-Economic
Routing of LLM Requests from People and Agents, with Evidence from Financial
Reports". It measures what it costs to resolve a
request when a router can choose among three tiers and can ask a clarifying
question first:

| Tier | Kind | Model |
| --- | --- | --- |
| local | self-hosted, existing owned hardware (priced at energy cost) | Llama 3.2 3B Instruct |
| private | self-hosted, rented GPU | Llama 3.1 8B Instruct |
| frontier | commercial API | Claude Sonnet 5.5 |

That gives six actions: answer at a tier, or ask and then answer at a tier.

Two experiments run separately, each into its own folder under `runs/`:

| Dataset | Items | Why |
| --- | --- | --- |
| `ambignq` | 2,000 open-domain questions, some ambiguous | general benchmark for clarification |
| `pacific` | 1,000 financial-report questions, half needing clarification | financial reports; the analysis also reweights to the natural 13% |

**To run it, follow `../docs/GPU_SETUP_AWS.md`.** It covers the AWS g6.2xlarge instance,
serving both self-hosted models with vLLM one after the other, the PACIFIC
download, and every command in order. This README explains the files and the
reproducibility rules.

## Files

| File | What it does |
| --- | --- |
| `config.yaml` | Tiers, models, prices, datasets, cost assumptions, analysis settings |
| `common.py` | Shared helpers: configuration, tiers, dataset folders, joining the data files |
| `datasets_io.py` | Loads AmbigNQ and PACIFIC; decides whether a response resolves the request (string match for AmbigNQ; numbers with tolerance and scale, or spans, for PACIFIC) |
| `collect.py` | `--stage clarify`: sample questions and write clarifying questions. `--stage answer --tier X`: one tier answers every question, as asked and after clarification |
| `pilot_check.py` | Summarizes a pilot: success rates, token counts, served model, sample answers, projected frontier cost |
| `profiles.py` | Derives the cost of a failed attempt and of asking for each caller profile (staff, agent) from the data, a BLS wage and published reading and typing rates |
| `reprofile.py` | Re-runs the analysis of a sealed bundle with the caller profiles; no model calls |
| `audit.py` | Draws a blind sample of responses for a manual check of the scorer, then reports agreement and error rates |
| `benchmark.py` | Throughput and GPU power of a self-hosted tier at increasing load |
| `kappa.py` | Cost per token of each self-hosted tier: rental price (private) or measured energy (local), at run-time and full load |
| `analyze.py` | Trains the router and compares routing policies; sensitivity analyses |
| `make_figures.py` | Draws Figure 2 from the result tables |
| `capture.py` | Seals a finished dataset run into a checksummed archive with a run record |
| `manifest.py` | Records hardware, software versions, models and prices at every step |

Every script takes `--dataset ambignq` or `--dataset pacific`. Files written
during a run, in `runs/<dataset>/`: `clarify.jsonl`, `answers_<tier>.jsonl`,
`benchmark_<tier>.json`, `kappa.json`, `config_kappa_*.yaml`,
`manifest_collect_*.json`, `results/`, `results_kappa_run/`.

## Reproducibility rules

1. **Clean environment.** Use a fresh virtual environment so the recorded
   package versions are exactly the ones used.
2. **Fill in `config.yaml`, then freeze it.** Use model commit hashes and exact
   API model names, never aliases. Save every pricing page you take a number
   from as a dated PDF. The config's hash is recorded, so later edits are visible.
3. **Record the PACIFIC commit hash** (`git rev-parse HEAD` in the clone).
4. **Pilot first** (20 questions per dataset), check the output by eye, then
   delete the pilot files so they never mix with the main run.
5. **One model on the GPU at a time.** For self-hosted tiers, `collect.py` and
   `benchmark.py` stop if the server reports a different model than the config
   names.
6. **Seal each run** with `capture.py` and write down the archive checksum. Keep
   one copy of each zip unchanged. To verify later, inside `run_<dataset>_<time>/`:

       sha256sum -c checksums.sha256      # macOS: shasum -a 256 -c checksums.sha256

## Notes

- Self-hosted tiers use vLLM's OpenAI-compatible server. The frontier tier uses
  the native Claude Messages API with `effort: low` and no temperature (Claude
  4.7 and later reject temperatures below 1). Any thinking it does is billed as
  output; the counts are saved per call and summed in the run record.
- If Hugging Face datasets are blocked on your network, download the AmbigNQ
  `train_light.json` from the AmbigQA project page and set
  `datasets.ambignq.local_json`.
- PACIFIC test answers are not public, so its sample is drawn from the train
  and validation splits.
- Temperature zero reduces variation but does not remove it. The saved data
  files are the record of the run: re-running `analyze.py` on them reproduces
  every number exactly; re-running collection reproduces the experiment, not
  identical responses.
- `delta`, `phi` and the local tier's unmeasured overheads are assumptions. The
  analysis sweeps each of them.
- `embedding_model: tfidf` exists to test the pipeline without downloading an
  encoder. Do not report results from it.
- Clarifying questions are checked: if the clarifier answers instead of asking, it
  is corrected up to twice, then a fixed neutral question is used. Both counts are
  in the run record and in `pilot_check.py`'s report.
- PACIFIC answers are scored on the final answer only (the last bold span, or the
  last paragraph after its last equals sign), so showing working earns no extra
  chances to match. Percentages and ratios are converted, and numbers are compared
  at the precision of either the answer or the gold value.
- Caller profiles: `policies_by_profile.csv`, `bootstrap_by_profile.csv` and, for
  PACIFIC, `policies_reweighted_by_profile.csv` give every policy under each profile.
  `profiles.json` records how each value was derived.
