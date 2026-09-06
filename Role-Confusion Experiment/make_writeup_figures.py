from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIRECTORY = Path(__file__).parent / "results"
FIGURE_DIRECTORY = RESULTS_DIRECTORY / "writeup_figures"
FIGURE_DIRECTORY.mkdir(exist_ok=True)


def save_clean_full_failures() -> None:
    summary_path = RESULTS_DIRECTORY / "gpt_oss_role_vector_clean_full_20260903_132212Z" / "summary.csv"
    summary = pd.read_csv(summary_path)
    labels = ["Baseline", "Fixed layer 12\nrole vector", "Layer-matched\nrole vector"]
    rates = summary["clear_injection_attempt_rate"].mul(100)
    colors = ["#c84d3c", "#2c7a7b", "#3a9d5d"]

    figure, axis = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    bars = axis.bar(labels, rates, color=colors, width=0.62)
    axis.set_ylabel("Clear behavioral failures (%)")
    axis.set_title("Role-vector interventions reduce injection-following failures")
    axis.set_ylim(0, max(rates) * 1.35)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, rate in zip(bars, rates):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{rate:.2f}%", ha="center", va="bottom")
    axis.text(0.5, -0.23, "633 injected prompts per condition; response-grounded classifier", ha="center", transform=axis.transAxes)
    figure.savefig(FIGURE_DIRECTORY / "clean_full_clear_behavioral_failures.png", dpi=220)
    plt.close(figure)


def save_layer_matched_userness_vs_failures() -> None:
    """Plot paired mechanistic and behavioral measurements from the layer-matched run."""
    run_id = "gpt_oss_role_vector_layer_matched_20260902_085528Z"
    with (RESULTS_DIRECTORY / run_id / "raw_results.json").open(encoding="utf-8") as file:
        raw_records = pd.DataFrame(json.load(file)["records"])
    injected_records = raw_records[raw_records["trial_type"].eq("injected")]
    mean_userness = injected_records.groupby("condition")["injection_userness_after"].mean()

    labels_path = RESULTS_DIRECTORY / "review_exports" / "all_runs_condition_reclassification_summary.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels["source_run_id"].eq(run_id)].set_index("condition")
    plot_data = pd.concat(
        [mean_userness.rename("mean_injection_userness"), labels["clear_behavioral_failure_rate"]],
        axis=1,
    ).reset_index()
    plot_data["vector_type"] = plot_data["condition"].str.extract(r"^(random_vector|layer_matched_role_vector)")
    plot_data["alpha"] = plot_data["condition"].str.extract(r"alpha_(\d+)$").astype(float)

    figure, axis = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    vector_styles = {
        "random_vector": ("Equal-norm random vector", "#8b6f47"),
        "layer_matched_role_vector": ("Layer-matched role vector", "#2c7a7b"),
    }
    label_offsets = {
        ("random_vector", 4): (10, 10),
        ("random_vector", 5): (10, -16),
        ("layer_matched_role_vector", 4): (10, 7),
        ("layer_matched_role_vector", 5): (10, 7),
    }
    for vector_type, (label, color) in vector_styles.items():
        series = plot_data[plot_data["vector_type"].eq(vector_type)].sort_values("alpha")
        axis.plot(
            series["mean_injection_userness"],
            series["clear_behavioral_failure_rate"],
            color=color,
            linewidth=1.4,
            zorder=2,
        )
        axis.scatter(
            series["mean_injection_userness"],
            series["clear_behavioral_failure_rate"],
            color=color,
            s=60,
            label=label,
            zorder=3,
        )
        for row in series.itertuples():
            offset = label_offsets[(vector_type, int(row.alpha))]
            axis.annotate(
                f"alpha={int(row.alpha)}",
                (row.mean_injection_userness, row.clear_behavioral_failure_rate),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8},
            )

    baseline = plot_data[plot_data["condition"].eq("baseline")].iloc[0]
    axis.scatter(
        baseline["mean_injection_userness"],
        baseline["clear_behavioral_failure_rate"],
        color="#c84d3c",
        s=65,
        label="Baseline",
        zorder=4,
    )
    axis.annotate(
        "baseline", (baseline["mean_injection_userness"], baseline["clear_behavioral_failure_rate"]),
        xytext=(5, 5), textcoords="offset points", fontsize=9,
    )
    axis.set_xlabel("Mean injection-token Userness after intervention")
    axis.set_ylabel("Clear behavioral-failure rate (%)")
    axis.set_title("Lower injection-token Userness accompanies fewer failures")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.5,
        -0.23,
        "Layer-matched run: 150 injected prompts per condition; response-grounded classifier",
        ha="center",
        transform=axis.transAxes,
    )
    figure.savefig(FIGURE_DIRECTORY / "layer_matched_userness_vs_clear_failures.png", dpi=220)
    plt.close(figure)


def save_baseline_userness_percentile_curve() -> None:
    """Show baseline failure rate across injected-command Userness percentile bins."""
    run_id = "gpt_oss_role_vector_20260829_054343Z"
    labels_path = RESULTS_DIRECTORY / "review_exports" / "all_runs_response_reclassification.csv"
    records = pd.read_csv(labels_path)
    baseline = records.loc[
        records["source_run_id"].eq(run_id)
        & records["trial_type"].eq("injected")
        & records["condition"].eq("baseline")
    ].copy()
    baseline["userness_percentile"] = baseline["injection_userness_before"].rank(pct=True, method="average") * 100
    baseline["percentile_bin"] = pd.qcut(baseline["userness_percentile"], q=20, labels=False)
    binned = baseline.groupby("percentile_bin", observed=True).agg(
        attacks=("trial_id", "size"),
        percentile_center=("userness_percentile", "mean"),
        failures=("clear_injection_attempt", "sum"),
        failure_rate=("clear_injection_attempt", "mean"),
    ).reset_index()
    binned["failure_rate"] *= 100
    assert len(baseline) == 600
    assert binned["attacks"].between(29, 31).all()

    figure, axis = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    axis.plot(
        binned["percentile_center"],
        binned["failure_rate"],
        color="#168fc5",
        marker="o",
        markersize=7,
        linewidth=2.0,
    )
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 25)
    axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"]) 
    axis.set_yticks([0, 5, 10, 15, 20, 25], ["0%", "5%", "10%", "15%", "20%", "25%"]) 
    axis.set_xlabel("Userness of injected command, percentile among baseline attacks")
    axis.set_ylabel("Clear behavioral-failure rate (%)")
    axis.set_title("Baseline injection failures by injected-command Userness")
    axis.grid(color="#d0d0d0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.5,
        -0.23,
        "Single-layer baseline: 600 injected prompts, 20 percentile bins (29-31 prompts each)",
        ha="center",
        transform=axis.transAxes,
    )
    figure.savefig(FIGURE_DIRECTORY / "baseline_userness_percentile_vs_clear_failures.png", dpi=220)
    plt.close(figure)


def save_random_vs_role_userness() -> None:
    """Compare injected-command Userness across random and role-vector magnitudes."""
    summary_path = RESULTS_DIRECTORY / "gpt_oss_role_vector_20260829_054343Z" / "summary.csv"
    summary = pd.read_csv(summary_path)
    baseline = summary.loc[summary["condition"].eq("baseline")].iloc[0]

    figure, axis = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    for prefix, label, color in [
        ("random_vector", "Equal-norm random vector", "#8b6f47"),
        ("role_vector", "Role vector", "#2c7a7b"),
    ]:
        series = summary.loc[summary["condition"].str.startswith(prefix)].copy()
        series["alpha"] = series["condition"].str.extract(r"alpha_(\d+)$").astype(float)
        series = series.sort_values("alpha")
        axis.plot(
            series["alpha"],
            series["mean_injection_userness_after"],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=label,
        )
    axis.axhline(
        baseline["mean_injection_userness_after"],
        color="#c84d3c",
        linewidth=1.4,
        linestyle="--",
        label="Baseline",
    )
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_xlabel("Intervention strength (alpha)")
    axis.set_ylabel("Mean injected-command Userness")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(FIGURE_DIRECTORY / "random_vs_role_injection_userness.png", dpi=220)
    plt.close(figure)


def save_random_vector_outcome_rates() -> None:
    """Plot random-vector behavioral and benign-tool-use rates in separate figures."""
    random_series = pd.DataFrame(
        {
            "alpha": [1, 2, 3, 4, 5],
            "clear_behavioral_failure_rate": [8.17, 7.67, 8.83, 8.50, 8.50],
            "benign_tool_use_rate": [80.67, 81.17, 81.00, 81.50, 81.50],
        }
    )
    baseline_rates = {
        "clear_behavioral_failure_rate": 9.00,
        "benign_tool_use_rate": 84.00,
    }

    plots = [
        ("clear_behavioral_failure_rate", "Clear behavioral-failure rate (%)", "#c84d3c", "random_vector_clear_failures.png"),
        ("benign_tool_use_rate", "Benign tool-use rate (%)", "#3a9d5d", "random_vector_benign_tool_use.png"),
    ]
    for column, y_label, color, filename in plots:
        figure, axis = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
        axis.plot(
            random_series["alpha"],
            random_series[column],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label="Equal-norm random vector",
        )
        axis.axhline(
            baseline_rates[column],
            color="#4b5563",
            linewidth=1.4,
            linestyle="--",
            label="Baseline",
        )
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xlabel("Intervention strength (alpha)")
        axis.set_ylabel(y_label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        figure.savefig(FIGURE_DIRECTORY / filename, dpi=220)
        plt.close(figure)


if __name__ == "__main__":
    save_clean_full_failures()
    save_layer_matched_userness_vs_failures()
    save_baseline_userness_percentile_curve()
    save_random_vs_role_userness()
    save_random_vector_outcome_rates()
    print(f"Wrote figures to {FIGURE_DIRECTORY}")