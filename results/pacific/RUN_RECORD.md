# Run record

Created 20261103T005727Z by capture.py. Every value below is read from the run's own files.

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
| Dataset | pacific (sampling seed 13) |
| Seeds | splits [1, 2, 3, 4, 5] |
| Thinking tokens billed, frontier tier | 206 |
| Clarifying questions: corrected once or twice / replaced by the fallback question | 0 / 0 of 1000 |
| Questions sampled | 1000 |
| Collection window, all tiers (UTC) | 2026-11-02T20:03:22.662+00:00 to 2026-11-02T21:24:15.684+00:00 |
| Calls retried after an error | 0 |
| Config used for the analysis | config_kappa_full_load.yaml, SHA-256 ed412c67c750ff7ead945229b739cf8994d0f85d2b619178163679eb7782c662 |

## Tiers

| Tier | Model requested | Model actually served (calls) | Calls | Tokens in / out | Prices in config (USD per M tokens, in / out) | Measured cost per token (USD per M) | Collection window (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| local (owned) | meta-llama/Llama-3.2-3B-Instruct (revision 0cb88a4f764b7a12671c53f0838cd831a0843b95) | meta-llama/Llama-3.2-3B-Instruct (3000) | 3,000 | 1,802,715 / 40,753 | 0.000204 / 0.000204 | run 0.0006; full load 0.0002 | 2026-11-02T20:03:22.662+00:00 to 2026-11-02T20:10:23.472+00:00 |
| private (rented) | meta-llama/Llama-3.1-8B-Instruct (revision 0e9e39f249a16976918f6564b8830bc894c89659) | meta-llama/Llama-3.1-8B-Instruct (2000) | 2,000 | 1,165,865 / 20,108 | 0.078071 / 0.078071 | run 0.0567; full load 0.0781 | 2026-11-02T21:20:07.898+00:00 to 2026-11-02T21:24:15.684+00:00 |
| frontier (api) | claude-sonnet-5-5 (revision None) | claude-sonnet-5-5 (2000) | 2,000 | 1,611,679 / 80,710 | 2.0 / 10.0 | n/a | 2026-11-02T20:40:35.908+00:00 to 2026-11-02T20:51:34.655+00:00 |

## Notes

AMI ami-024b2999242b4b38d; PACIFIC commit 66a52ad4fd9eb7ad71e82a5c0bfb1de226366827; vLLM 0.33.0; instance g6.2xlarge; clarifier 0 corrections and 0 fallbacks of 3000; adds prior-corrected router robustness check; re-analysed from sealed bundle run_pacific_20261103T001644Z with caller profiles (profiles.py), no new model calls

## Checks before submitting

- Fields still marked TODO: none
- More than one served version for a tier means the model changed during the run;
  report both and say so in Section 5.2.
- Exact package versions are in environment.txt.
