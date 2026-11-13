# Running the experiment on AWS g6.2xlarge (three tiers, two datasets)

One AWS instance with one NVIDIA L4 GPU serves both self-hosted tiers, one
after the other, through vLLM. The frontier tier is Claude Sonnet 5.5 through
the Claude API. Each dataset runs into its own folder, `runs/ambignq/` and
`runs/pacific/`.

| Tier | Model | Cost basis |
| --- | --- | --- |
| local | `meta-llama/Llama-3.2-3B-Instruct` | existing owned hardware: marginal energy cost (measured GPU power x electricity price) |
| private | `meta-llama/Llama-3.1-8B-Instruct` | rented: the instance's hourly price |
| frontier | `claude-sonnet-5-5` | Claude API posted prices |

Facts checked on 2 November 2026, the day of the reported run (confirm on the day you run):

- g6.2xlarge: 1x NVIDIA L4 (22 GiB GPU memory), 8 vCPUs, 32 GiB RAM,
  $0.9776 per hour on demand in us-east-1. This is the instance used for the
  results in the paper. A g6.xlarge (4 vCPUs, 16 GiB RAM) has the same GPU and
  costs less, but was not used for the reported run.
- The AWS "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)"
  supports G6 and ships the NVIDIA driver, CUDA 12.8 and Python 3.12. It is x86_64.
- vLLM supports the L4 and Python 3.11 to 3.14.
- Claude Sonnet 5.5 costs $2 / $10 per million input / output tokens. The
  scripts call the native Claude API with effort set to low and no temperature
  (Claude 4.7 and later reject temperatures below 1).

Commands marked `[laptop]` run on your Mac; `[instance]` on the AWS machine.
Expect one to two hours of setup, then several hours of collection.

---

## 0. Before the day of the run

1. **GPU quota.** *Service Quotas → Amazon EC2 → Running On-Demand G and VT
   instances*: at least 8 vCPUs in your region for g6.2xlarge.
   Approval can take a day or more.
2. **Llama access.** Both models are gated. On Hugging Face, open each model
   page, accept Meta's licence, and create a read access token.
3. **Model versions.** On each model page open *Files and versions*, copy the
   latest commit hash into `config.yaml` under that tier's `revision`.
4. **Claude API key.** Create a key in the Claude Console. Save the Claude
   pricing page as a dated PDF.
5. **Instance price.** Find g6.2xlarge on the AWS EC2 on-demand pricing page,
   put the hourly price in `hardware.gpu_hourly_usd`, and save a dated PDF.
6. **Local tier cost.** Nothing to supply: the electricity price is in
   `config.yaml` and the GPU's power is measured by `benchmark.py`.

## 1. Launch the instance

`[laptop]` Find the latest AMI id (AWS CLI configured):

    aws ssm get-parameter --region us-east-1 \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
      --query "Parameter.Value" --output text

In the EC2 console choose *Launch instance*:

- **AMI:** the id above (architecture 64-bit x86)
- **Instance type:** g6.2xlarge
- **Key pair:** yours
- **Network:** SSH (port 22) **from your IP only**; do not open port 8000
- **Storage:** 120 GB gp3

Note the AMI id; it goes in the paper.

`[laptop]`

    ssh -i your-key.pem ubuntu@<instance-public-ip>

`[instance]`

    nvidia-smi        # expect one NVIDIA L4 with about 23 GB

## 2. Install software and data

`[instance]`

    sudo apt-get update && sudo apt-get install -y tmux unzip git
    curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
    uv venv ~/venv --python 3.12 && source ~/venv/bin/activate
    uv pip install vllm --torch-backend=auto

`[instance]` get the code:

    git clone https://github.com/<github-user>/llm-routing-cost-per-resolution.git
    cd llm-routing-cost-per-resolution/code
    uv pip install -r requirements.txt
    git clone https://github.com/dengyang17/PACIFIC.git      # PACIFIC data (MIT licence)
    (cd PACIFIC && git rev-parse HEAD)                       # note this commit hash

Fill in `config.yaml` in this folder (or copy your own over it). Its
`datasets.pacific.path` already points to `PACIFIC/data/pacific`.

## 3. Common settings for every shell

Each tmux window needs these (tmux: `Ctrl-b c` new window, `Ctrl-b n` next,
`Ctrl-b d` detach, `tmux attach` to return):

    source ~/venv/bin/activate && cd ~/llm-routing-cost-per-resolution/code
    export SMALL_API_KEY=<any value you choose, the same in every window>
    export ANTHROPIC_API_KEY=<your Claude API key>
    export HF_TOKEN=<your Hugging Face token>

## 4. Local tier (Llama 3.2 3B), both datasets

Window 1, start the server and keep it running. The longer context setting is
for PACIFIC's report tables:

    tmux new -s run
    # common settings from step 3, then:
    vllm serve meta-llama/Llama-3.2-3B-Instruct --revision <local revision> \
      --host 127.0.0.1 --port 8000 --api-key "$SMALL_API_KEY" \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 \
      --seed 0 2>&1 | tee vllm_local.log

Window 2, once the server says it is running. **Pilot first** (20 questions per
dataset):

    for DS in ambignq pacific; do
      python collect.py --dataset $DS --stage clarify --limit 20
      python collect.py --dataset $DS --stage answer --tier local --limit 20
      python collect.py --dataset $DS --stage answer --tier frontier --limit 20
    done

Check the output with `python pilot_check.py --dataset ambignq` and
`python pilot_check.py --dataset pacific`: clarifying questions are real
questions, answers are short, token counts are non-zero, the served model is
right, and for the frontier tier thinking tokens are small. It also projects the
full frontier cost. Send me its output. Then remove the pilot data:

    rm -r runs

Full run for this tier, then its throughput benchmark, for each dataset:

    for DS in ambignq pacific; do
      python collect.py --dataset $DS --stage clarify --workers 16
      python collect.py --dataset $DS --stage answer --tier local --workers 16
      python benchmark.py --dataset $DS --tier local
    done

`--workers` only sets how many requests are in flight; it speeds up collection
and does not change token counts or costs (throughput is measured separately by
`benchmark.py`). Keep the frontier tier at the default of 4 to stay inside the
API's rate limits. Do not run anything else on the GPU meanwhile. Each command can be rerun after
an interruption and continues where it stopped.

## 5. Frontier tier (any time after the clarify stage)

It uses only the Claude API, so run it in a third window while the GPU works:

    for DS in ambignq pacific; do python collect.py --dataset $DS --stage answer --tier frontier; done

## 6. Private tier (Llama 3.1 8B), both datasets

In window 1 stop the 3B server (`Ctrl-C`) and start the 8B model:

    vllm serve meta-llama/Llama-3.1-8B-Instruct --revision <private revision> \
      --host 127.0.0.1 --port 8000 --api-key "$SMALL_API_KEY" \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 \
      --seed 0 2>&1 | tee vllm_private.log

Window 2:

    for DS in ambignq pacific; do
      python collect.py --dataset $DS --stage answer --tier private --workers 16
      python benchmark.py --dataset $DS --tier private
    done

If the 3B model is still loaded, the script stops with "server is serving ...
restart vLLM with the right model" and writes nothing. Then stop the server.

## 7. Costs per token, analysis and sealing, per dataset

With `hardware.gpu_hourly_usd` filled in:

    for DS in ambignq pacific; do
      python kappa.py --dataset $DS
      python profiles.py --dataset $DS --config config.yaml runs/$DS/config_kappa_full_load.yaml runs/$DS/config_kappa_run.yaml --write
      python analyze.py --dataset $DS
      python analyze.py --dataset $DS --config runs/$DS/config_kappa_run.yaml --out runs/$DS/results_kappa_run
      python capture.py --dataset $DS --notes "AMI ami-xxxx; PACIFIC commit xxxx; anything unusual"
    done

Each `capture.py` prints an archive checksum; write both down.

## 8. Copy home and shut down

`[laptop]`

    scp -i your-key.pem "ubuntu@<instance-public-ip>:~/llm-routing-cost-per-resolution/code/runs/*/run_*.zip" .

Check both zips open, then **terminate** the instance in the EC2 console (a
stopped instance still bills for its disk). Note the time.

## What goes in the paper from this run

| Item | Where it comes from |
| --- | --- |
| Instance, region, GPU, AMI | `config.yaml`, manifests (nvidia-smi), your note |
| Hourly price and date | `config.yaml`, saved pricing PDF |
| Local tier energy cost | `kappa.json` (measured GPU power), electricity price in `config.yaml` |
| Models and commit hashes; Claude model served | `config.yaml`, `RUN_RECORD.md` |
| PACIFIC data version | the commit hash noted in step 2 |
| vLLM and library versions | `environment.txt`, manifests |
| Serving settings | the `vllm serve` commands above, `vllm_*.log` |
| Cost per token by tier, run and full load | `runs/<dataset>/kappa.json` |
| Throughput by concurrency | `runs/<dataset>/benchmark_*.json` |
| Claude thinking tokens billed | `RUN_RECORD.md` |
