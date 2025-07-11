"""
Evaluates results from ISIC 2019 fine-tuning and linear probing experiments.
Conducts paired t-tests to assess statistical significance across resolutions
and model variations. For single runs, skips t-tests and
summarizes perform ance. Generates plots and saves results to JSON.
"""
# pylint: disable=broad-exception-caught

import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel


def load_results(
    finetune_file="results_metrics_finetune.json",
    linear_probe_file="results_metrics_linear_probe.json",
):
    """
    Loads fine-tuning and linear probing results from JSON files, supporting
    multiple runs or single run per condition.
    """
    results = {}
    for file, mode in [
        (finetune_file, "finetune"),
        (linear_probe_file, "linear_probe"),
    ]:
        if not Path(file).exists():
            print(f"Warning: {file} not found, skipping {mode} results.")
            continue
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model, qualities in data.items():
            for quality, metrics in qualities.items():
                key = (model, int(quality), mode)
                eval_metrics = metrics["eval_metrics"]
                # Handle single run (dict) or multiple runs (list of dicts)
                if isinstance(eval_metrics, dict):
                    results[key] = [eval_metrics]  # Wrap single run in list
                elif isinstance(eval_metrics, list):
                    results[key] = eval_metrics
                else:
                    print(f"Warning: Invalid eval_metrics format for {key}, skipping.")
                    continue
    return results


def paired_t_tests(metrics, models, qualities, modes):
    """
    Conducts paired t-tests across models and modes if multiple
    runs are available.
    """
    t_test_results = []
    metric_names = ["accuracy", "f1", "auc"]

    def run_if_valid(k1, k2, label):
        if (
            k1 in metrics and k2 in metrics and
            len(metrics[k1]) == len(metrics[k2]) > 1
        ):
            for metric in metric_names:
                try:
                    v1 = [r[metric] for r in metrics[k1]]
                    v2 = [r[metric] for r in metrics[k2]]
                    stat, p = ttest_rel(v1, v2)
                    t_test_results.append({
                        "comparison": label.format(metric=metric),
                        "statistic": stat,
                        "p_value": p,
                    })
                except Exception as e:
                    print(f"Error in t-test for {label.format(metric=metric)}: {e}")

    has_multiple_runs = any(len(r) > 1 for r in metrics.values())
    if not has_multiple_runs:
        print("Warning: Only single-run metrics available. Skipping t-tests.")
        return t_test_results

    # 1. Within-model: JPEG quality comparisons
    for model, mode in itertools.product(models, modes):
        for q1, q2 in itertools.combinations(qualities, 2):
            run_if_valid(
                (model, q1, mode),
                (model, q2, mode),
                f"{model}_{mode}" + "_{{metric}}_jpeg{q1}_vs_jpeg{q2}"
            )

    # 2. Across models: same quality & mode
    for m1, m2 in itertools.combinations(models, 2):
        for quality, mode in itertools.product(qualities, modes):
            run_if_valid(
                (m1, quality, mode),
                (m2, quality, mode),
                f"{m1}_vs_{m2}_{mode}" + "_{{metric}}_jpeg{quality}"
            )

    # 3. Finetune vs. Linear Probe: same model & quality
    for model, quality in itertools.product(models, qualities):
        run_if_valid(
            (model, quality, "finetune"),
            (model, quality, "linear_probe"),
            f"{model}_finetune_vs_linear_probe" + "_{{metric}}_jpeg{quality}"
        )

    return t_test_results


def summarize_performance(metrics):
    """
    Summarizes performance metrics (mean, std) across models, qualities, and modes.
    """
    summary = []
    for (model, quality, mode), runs in metrics.items():
        if not runs:
            continue

        # Compute mean and std for each metric
        accuracies = [run.get("accuracy") for run in runs if "accuracy" in run]
        f1s = [run.get("f1") for run in runs if "f1" in run]
        aucs = [run.get("auc") for run in runs if "auc" in run]

        summary.append(
            {
                "model": model,
                "mode": mode,
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies, ddof=1) if len(runs) > 1 else 0,
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s, ddof=1) if len(runs) > 1 else 0,
                "auc_mean": np.mean(aucs),
                "auc_std": np.std(aucs, ddof=1) if len(runs) > 1 else 0,
            }
        )
    return pd.DataFrame(summary)


def plot_performance(df):
    """Generates plots for performance metrics across models."""
    metrics = ["accuracy", "f1", "auc"]
    modes = ["finetune", "linear_probe"]

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for mode in modes:
            subset = df[df["mode"] == mode]
            sns.lineplot(
                x="jpeg_quality",
                y=f"{metric}_mean",
                hue="model",
                style="model",
                markers=True,
                dashes=False,
                data=subset,
                label=f"{mode}",
            )
        plt.title(f"{metric.capitalize()} vs. JPEG Quality")
        plt.xlabel("JPEG Quality")
        plt.ylabel(f"{metric.capitalize()} Mean")
        plt.legend()
        plt.savefig(f"{metric}_vs_jpeg_quality.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Bar plot for model comparison
    for mode in modes:
        plt.figure(figsize=(12, 6))
        subset = df[df["mode"] == mode]
        subset_melted = subset.melt(
            id_vars=["model", "jpeg_quality", "mode"],
            value_vars=["accuracy_mean", "f1_mean", "auc_mean"],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(
            x="jpeg_quality",
            y="value",
            hue="model",
            style="metric",
            data=subset_melted,
        )
        plt.title(f"Performance Metrics ({mode.capitalize()})")
        plt.xlabel("JPEG Quality")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig(f"metrics_{mode}_bar.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main function to evaluate results and conduct statistical tests."""
    # Load results
    metrics = load_results()

    if not metrics:
        print("No results found. Please ensure JSON files exist.")
        return

    # Conduct paired t-tests (if multiple runs)
    # Dynamic model/quality/mode extraction
    models = sorted(set(k[0] for k in metrics))
    qualities = sorted(set(k[1] for k in metrics))
    modes = sorted(set(k[2] for k in metrics))
    t_test_results = paired_t_tests(metrics, models, qualities, modes)

    # Summarize performance
    performance_df = summarize_performance(metrics)

    # Generate plots
    plot_performance(performance_df)

    # Save results
    results = {
        "t_test_results": t_test_results,
        "performance_summary": performance_df.to_dict(orient="records"),
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete. Results saved to 'evaluation_results.json'.")
    print(
        "Plots saved: accuracy_vs_jpeg_quality.png, f1_vs_jpeg_quality.png, "
        "auc_vs_jpeg_quality.png, metrics_finetune_bar.png, metrics_linear_probe_bar.png"
    )


if __name__ == "__main__":
    main()
