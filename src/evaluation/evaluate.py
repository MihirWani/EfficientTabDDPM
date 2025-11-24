# scripts/evaluate_all.py
"""
Orchestrator: compute distributional, structural, and utility metrics in one run.
Usage example:
python scripts/evaluate_all.py \
  --real_csv data/processed/adult/test.csv \
  --synth_csv results/samples/adult_generated_epoch1.csv \
  --preprocessor data/processed/adult/preprocessor \
  --target income \
  --run_name adult_epoch1_test
"""
import argparse
import os
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation import metrics_distribution as mdist
from src.evaluation import metrics_structure as mstruct
from src.evaluation import metrics_utility as muti

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_csv", type=str, required=True)
    p.add_argument("--synth_csv", type=str, required=True)
    p.add_argument("--preprocessor", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--run_name", type=str, default="eval_run")
    p.add_argument("--resource_pdf", type=str, default="/mnt/data/EE782 2025 A1 Image Captioning.pdf")
    return p.parse_args()

def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def main():
    args = parse_args()
    out_dir = Path("results/eval") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.real_csv)
    synth = pd.read_csv(args.synth_csv)

    # load preprocessor metadata to know numeric/categorical columns
    meta = json.load(open(Path(args.preprocessor)/"metadata.json","r"))
    numeric_cols = meta.get("numeric", [])
    cat_cols = meta.get("categorical", [])

    # DISTRIBUTIONAL
    dist_report = mdist.distribution_summary(real, synth, numeric_cols, cat_cols)
    save_json(dist_report, out_dir/"distribution_metrics.json")

    # simple plots: numeric marginal overlays for top 6 numeric cols by variance
    try:
        vars_sorted = sorted(numeric_cols, key=lambda c: real[c].var() if c in real.columns else 0.0, reverse=True)[:6]
        for col in vars_sorted:
            plt.figure(figsize=(6,3))
            sns.kdeplot(real[col].dropna(), label="real", fill=False)
            sns.kdeplot(synth[col].dropna(), label="synth", fill=False)
            plt.title(f"{col} marginal")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir/f"marginal_{col}.png")
            plt.close()
    except Exception as e:
        print("Plotting marginals failed:", e)

    # CATEGORICAL bar plots (top 6)
    try:
        for col in cat_cols[:6]:
            plt.figure(figsize=(6,3))
            r = real[col].astype(str).fillna("##NA##").value_counts(normalize=True).sort_values(ascending=False)[:20]
            s = synth[col].astype(str).fillna("##NA##").value_counts(normalize=True).sort_values(ascending=False)[:20]
            dfp = pd.concat([r.rename("real"), s.rename("synth")], axis=1).fillna(0)
            dfp.plot.bar(figsize=(8,3))
            plt.title(f"{col} freq (top categories)")
            plt.tight_layout()
            plt.savefig(out_dir/f"catfreq_{col}.png")
            plt.close()
    except Exception as e:
        print("Categorical plotting failed:", e)

    # STRUCTURAL
    struct_report = {}
    struct_report.update(mstruct.correlation_distance(real, synth, numeric_cols))
    struct_report.update(mstruct.mutual_information_matrix(real, synth, numeric_cols, cat_cols))
    struct_report.update(mstruct.coverage_diversity(real, synth, numeric_cols))
    save_json(struct_report, out_dir/"structural_metrics.json")

    # UTILITY
    feature_cols = [c for c in (numeric_cols + cat_cols) if c in real.columns and c in synth.columns and c != args.target]
    # For utility we need numeric-only or encoded features; for simplicity use numeric_cols only
    utility_report = muti.utility_evaluation(real, synth, features=numeric_cols, target_col=args.target)
    save_json(utility_report, out_dir/"utility_metrics.json")

    # summary JSON aggregating all
    summary = {
        "distribution": dist_report,
        "structural": struct_report,
        "utility": utility_report,
        "resource_pdf": args.resource_pdf
    }
    save_json(summary, out_dir/"full_report.json")
    print("Evaluation completed. Results saved to:", out_dir)

if __name__ == "__main__":
    main()
