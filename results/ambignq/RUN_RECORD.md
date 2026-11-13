# Run record

Created 20261103T005642Z by capture.py. Every value below is read from the run's own files.

## Hardware and software (Section 5.7)

| Item | Value |
| --- | --- |
| Compute provider and instance | Amazon Web Services (EC2) / g6.2xlarge, us-east-1, 1x NVIDIA L4, 8 vCPU, 32 GiB |
| GPU rental price per hour (USD) | 0.9776 |
| GPU detected during collection | NVIDIA L4, 23034 MiB, 595.112.07 |
| Local tier cost basis | marginal energy on existing hardware; power measured GPU draw; electricity 0.1341 USD/kWh; other 0 USD/h |
| Analysis machine | macOS-26.5.1-arm64-arm-64bit; 10 cores; 32.0 GB RAM |
| Python | 3.12.16 |
| Sentence encoder | sentence-transformers/all-MiniLM-L6-v2 |
| Dataset | ambignq (sampling seed 13) |
| Seeds | splits [1, 2, 3, 4, 5] |
| Thinking tokens billed, frontier tier | 48,327 |
| Clarifying questions: corrected once or twice / replaced by the fallback question | 0 / 0 of 2000 |
| Questions sampled | 2000 |
| Collection window, all tiers (UTC) | 2026-11-02T20:01:04.968+00:00 to 2026-11-02T21:16:53.156+00:00 |
| Calls retried after an error | 0 |
| Config used for the analysis | config_kappa_full_load.yaml, SHA-256 a9c12682c00043b4aa6f4ee3cb3757aab56c151596978f8211eb184e5c4023b0 |

## Tiers

| Tier | Model requested | Model actually served (calls) | Calls | Tokens in / out | Prices in config (USD per M tokens, in / out) | Measured cost per token (USD per M) | Collection window (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| local (owned) | meta-llama/Llama-3.2-3B-Instruct (revision 0cb88a4f764b7a12671c53f0838cd831a0843b95) | meta-llama/Llama-3.2-3B-Instruct (6000) | 6,000 | 651,163 / 79,099 | 0.000358 / 0.000358 | run 0.0013; full load 0.0004 | 2026-11-02T20:01:04.968+00:00 to 2026-11-02T20:07:05.957+00:00 |
| private (rented) | meta-llama/Llama-3.1-8B-Instruct (revision 0e9e39f249a16976918f6564b8830bc894c89659) | meta-llama/Llama-3.1-8B-Instruct (4000) | 4,000 | 332,610 / 29,616 | 0.098752 / 0.098752 | run 0.1312; full load 0.0988 | 2026-11-02T21:13:58.155+00:00 to 2026-11-02T21:16:53.156+00:00 |
| frontier (api) | claude-sonnet-5-5 (revision None) | claude-sonnet-5-5 (4000) | 4,000 | 292,308 / 223,406 | 2.0 / 10.0 | n/a | 2026-11-02T20:15:25.730+00:00 to 2026-11-02T20:40:33.470+00:00 |

## Notes

AMI ami-024b2999242b4b38d; PACIFIC commit 66a52ad4fd9eb7ad71e82a5c0bfb1de226366827; vLLM 0.33.0; instance g6.2xlarge; clarifier 0 corrections and 0 fallbacks of 3000; adds prior-corrected router robustness check; re-analysed from sealed bundle run_ambignq_20261103T001524Z with caller profiles (profiles.py), no new model calls

## Checks before submitting

- Fields still marked TODO: none
- More than one served version for a tier means the model changed during the run;
  report both and say so in Section 5.2.
- Exact package versions are in environment.txt.
