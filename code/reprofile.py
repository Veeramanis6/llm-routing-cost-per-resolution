"""Re-run the analysis of a sealed run with the caller profiles (staff and agent).

No model is called. The data are copied unchanged from the sealed bundle; the
updated analysis code is taken from the folder this script sits in; profiles.py
derives delta and phi for each profile from the data and writes them into the
configuration; the analysis runs and a new bundle is sealed.

    python reprofile.py --bundle ~/token_router_rerun/runs/pacific/run_pacific_20261102T225437Z \
        --work ~/token_router_profiles --notes "..."
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

UPDATED = ["analyze.py", "capture.py", "profiles.py", "reprofile.py", "README.md"]
SKIP = {"code", "results", "results_kappa_run", "RUN_RECORD.md", "checksums.sha256",
        "environment.txt", "question_ids.txt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="an unzipped run_<dataset>_<time> folder")
    parser.add_argument("--work", required=True, help="working folder (one per run of this script)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--embedding-model", default=None, help="testing only")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    bundle = Path(args.bundle).expanduser().resolve()
    dataset = bundle.name.split("_")[1]
    work = Path(args.work).expanduser().resolve()
    data = work / "runs" / dataset
    if data.exists():
        raise SystemExit(f"{data} already exists; use a new --work folder or remove it")
    data.mkdir(parents=True)
    for f in (bundle / "code").iterdir():
        shutil.copy2(f, work / f.name)
    for name in UPDATED:
        if (here / name).exists():
            shutil.copy2(here / name, work / name)
    for f in bundle.iterdir():
        if f.name not in SKIP and f.is_file():
            shutil.copy2(f, (work / f.name) if f.name == "config.yaml" else (data / f.name))

    configs = ["config.yaml", f"runs/{dataset}/config_kappa_full_load.yaml", f"runs/{dataset}/config_kappa_run.yaml"]
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=work, check=True)
    run("profiles.py", "--dataset", dataset, "--config", *configs, "--write")
    if args.embedding_model:
        import yaml
        for c in configs:
            d = yaml.safe_load(open(work / c, encoding="utf-8"))
            d["router"]["embedding_model"] = args.embedding_model
            (work / c).write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    run("analyze.py", "--dataset", dataset)
    run("analyze.py", "--dataset", dataset, "--config", f"runs/{dataset}/config_kappa_run.yaml",
        "--out", f"runs/{dataset}/results_kappa_run")
    notes = f"{args.notes}; re-analysed from sealed bundle {bundle.name} with caller profiles (profiles.py), no new model calls".lstrip("; ")
    run("capture.py", "--dataset", dataset, "--notes", notes)


if __name__ == "__main__":
    main()
