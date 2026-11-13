# llm-routing-cost-per-resolution

Code, data and materials for the article

> **Answer, Ask or Escalate: Token-Economic Routing of LLM Requests from People and Agents, with Evidence from Financial Reports**
> &lt;Author Name&gt;. Submitted to the *Journal of Artificial Intelligence Research*, 2026.

The article treats an LLM gateway as an economic decision. For every request, the
gateway can answer with a small model on hardware the organization already owns,
answer with a mid-size model on rented GPUs, answer with a commercial frontier
model, or first ask the user a clarifying question and then answer at any of the
three tiers. Each of these six actions is judged by one yardstick: the expected
cost per resolved request. That cost includes the tokens and also two costs that
no invoice shows. The first, δ, is the cost of a failed attempt to whoever sent
the request. The second, φ, is the friction of being asked a question. The values
of δ and φ depend on who the caller is (a member of staff, or an AI agent that
does or does not notice a failure), and so does the cheapest policy.

This repository contains everything needed to check the numbers in the article,
re-price them under other assumptions, or run the experiment again.

| Tier | Model | Where it ran | Cost basis |
| --- | --- | --- | --- |
| local | Llama 3.2 3B Instruct | owned GPU | measured marginal energy |
| private | Llama 3.1 8B Instruct | rented AWS g6.2xlarge, 1x NVIDIA L4 | hourly rental price |
| frontier | Claude Sonnet 5.5 | Claude API | posted token prices |

| Dataset | Questions | Role |
| --- | --- | --- |
| AmbigNQ | 2,000 | open-domain questions, some ambiguous |
| PACIFIC | 1,000 | questions over company financial reports, half needing clarification; also reweighted to the natural 13% |

## Repository layout

```
code/        the full pipeline (collection, cost measurement, router, analysis, figures)
docs/        step-by-step guide to running the experiment on AWS
data/raw/    model responses and run files from the reported run (see data/raw/README.md)
results/     every result table of the reported run, as CSV and as one workbook per dataset
audit/       the blind manual audit of the automatic scorer (100 AmbigNQ responses)
figures/     Figures 1 and 2 (PNG, PDF, Mermaid source, editable PowerPoint, chart data)
```

### `results/`

`results/<dataset>/results/` holds the main results, with self-hosted tiers costed
at full load. `results/<dataset>/results_kappa_run/` holds the same tables with
costs measured during the run (the run-time basis). Each folder also has
`RUN_RECORD.md` (hardware, models, model revisions, token counts, collection
times), `kappa.json` (measured cost per token of each tier) and `profiles.json`
(how δ and φ were derived for each caller profile).

`results_AmbigNQ.xlsx` and `results_PACIFIC.xlsx` contain the same tables, one
sheet per table, with a README sheet. Sheets whose names start with `rt_` use the
run-time cost basis.

| File | Contents | Article |
| --- | --- | --- |
| `descriptive.csv` | success rate and mean cost of each of the six actions | Section 6.2 |
| `estimator_quality.csv` | AUC and Brier score of the success estimators | Section 6.3 |
| `policies_by_profile.csv` | every routing policy under each caller profile | Section 6.4, Figure 2 |
| `bootstrap_by_profile.csv` | paired bootstrap of the router against the baselines | Section 6.4 |
| `policies_by_ambiguity.csv` | every policy on ambiguous and clear questions separately | supporting table |
| `policies_reweighted*.csv` | PACIFIC reweighted to its natural 13% ambiguity rate | Section 6.5 |
| `sensitivity_*.csv` | one assumption varied at a time: δ, φ, API prices, utilization, owned-hardware overheads | Section 6.6 |

The blind manual audit of the scorer (`audit/`) is reported in Section 6.2.

## Reproducing the results

There are three levels, from quickest to most complete.

**1. Check a number.** Every figure in Sections 6 and 7 of the article can be
read directly from the CSV files in `results/` or from the workbooks.

**2. Re-run the analysis without calling any model.** This rebuilds every table
from the saved responses. It runs on a laptop in a few minutes per dataset and
reproduces the reported numbers exactly.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
mkdir -p code/runs && cp -r data/raw/ambignq data/raw/pacific code/runs/
cd code
for DS in ambignq pacific; do
  python analyze.py --dataset $DS
  python analyze.py --dataset $DS --config runs/$DS/config_kappa_run.yaml --out runs/$DS/results_kappa_run
done
python make_figures.py runs ../figures
```

To re-price the results under your own assumptions, change the `economics`
block of `runs/<dataset>/config_kappa_full_load.yaml`, or run `profiles.py`
with your own wages or correction time, and run `analyze.py` again.

**3. Run the whole experiment again.** Follow `docs/GPU_SETUP_AWS.md`. You will
need a Hugging Face token with access to the two Llama models, a Claude API key
and an AWS account with quota for one L4 GPU. Expect a few hours of GPU time and
a few dollars of API cost. Temperature zero reduces variation between runs but
does not remove it, so a new run reproduces the experiment rather than identical
responses.

## Data and licences

- **Code** (`code/` and the Mermaid source `figures/figure1_gateway.mmd`): MIT licence, see `LICENSE`.
- **Data, results, audit and figures produced for this article**: CC BY 4.0, see `LICENSE-DATA`.
- **AmbigNQ** questions and answers come from AmbigQA (Min et al., 2020), released under CC BY-SA 3.0.
  Files in `data/raw/ambignq/` that contain question or answer text keep that licence.
- **PACIFIC** questions and report excerpts come from the PACIFIC repository (Deng et al., 2022),
  released under the MIT licence. Files in `data/raw/pacific/` that contain this text keep that licence.
- **Model outputs** were produced with Llama 3.2 3B Instruct and Llama 3.1 8B Instruct (Meta Llama
  licences and acceptable-use policies) and with Claude Sonnet 5.5 (Anthropic). They are released
  for research and reproduction.

No personal or customer data are included. The financial questions come from
published company reports.

## Citation

If you use this code or data, please cite the article. `CITATION.cff` has the
details, and GitHub shows them under "Cite this repository".

## Contact

&lt;Author Name&gt;, &lt;affiliation&gt; — &lt;email&gt;. Questions and issues are welcome through GitHub Issues.
