#!/usr/bin/env python3
"""Run Adaptive SCJKE-PCM on DrugBAN BindingDB and BioSNAP cluster splits."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="scjke_pcm/results")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "scjke_pcm.py"

    results = {}
    for dataset in ("bindingdb", "biosnap"):
        source = root / "datasets" / dataset / "cluster" / "source_train.csv"
        query = root / "datasets" / dataset / "cluster" / "target_test.csv"
        predictions = output / f"{dataset}_cluster_predictions.csv"
        metrics = output / f"{dataset}_cluster_metrics.json"
        command = [
            sys.executable,
            str(script),
            "--source",
            str(source),
            "--query",
            str(query),
            "--output",
            str(predictions),
            "--metrics",
            str(metrics),
        ]
        subprocess.run(command, check=True)
        results[dataset] = json.loads(metrics.read_text(encoding="utf-8"))

    summary_path = output / "cluster_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved combined metrics to {summary_path}")


if __name__ == "__main__":
    main()
