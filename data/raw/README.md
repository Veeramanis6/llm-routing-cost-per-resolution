# Raw data from the reported run

This folder holds the files the analysis reads, one subfolder per dataset
(`ambignq/` and `pacific/`), plus the two sealed run bundles as zip files:

    data/raw/run_ambignq_20261103T005642Z.zip
    data/raw/run_pacific_20261103T005727Z.zip
    data/raw/ambignq/   the data files from the AmbigNQ bundle (list below)
    data/raw/pacific/   the data files from the PACIFIC bundle

Files in each dataset subfolder:

| File | Contents |
| --- | --- |
| `clarify.jsonl` | the sampled questions, the intended reading, the accepted answers and the clarifying question written for each |
| `answers_local.jsonl`, `answers_private.jsonl`, `answers_frontier.jsonl` | one line per question and tier: the response as asked and after clarification, token counts, served model, timestamps |
| `benchmark_local.json`, `benchmark_private.json` | throughput and GPU power of each self-hosted tier at increasing load |
| `kappa.json` | measured cost per token of each tier |
| `config_kappa_full_load.yaml`, `config_kappa_run.yaml` | the configurations the analysis used, with measured prices and caller profiles filled in |
| `manifest_collect_*.json` | hardware and software recorded at each collection step |
| `profiles.json` | how δ and φ were derived for each caller profile |

The two sealed run bundles, `run_ambignq_20261103T005642Z.zip` and
`run_pacific_20261103T005727Z.zip`, are the record of the run reported in the
article. Each contains the files above, the code that produced them, the run
record and a SHA-256 checksum for every file. To verify a bundle after unzipping
it:

    cd run_<dataset>_<time>
    sha256sum -c checksums.sha256        # macOS: shasum -a 256 -c checksums.sha256

## Licences

Question, answer and report text in these files comes from AmbigQA (CC BY-SA 3.0)
and PACIFIC (MIT licence) and keeps those licences. Everything else here was
produced for the article and is released under CC BY 4.0 (`LICENSE-DATA`).
