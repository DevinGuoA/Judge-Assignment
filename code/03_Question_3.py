### Setup ###

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "PhdFinance_FNCE7020_HW1_Data.csv"
OUTPUT_DIR = ROOT / "output"
TMP_DIR = ROOT / "tmp" / "results"
CACHE_DIR = TMP_DIR / "cache"
MPL_DIR = TMP_DIR / "matplotlib"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

treatment = "Treatment_Judge"
industry = "Char5_FirmIndustry"
characteristics = {
    "Char1_FirmValueBeforeFiling": "Firm value",
    "Char2_FirmProfitMarginBeforeFiling": "Profit margin",
    "Char3_FirmInterestCoverageBeforeFiling": "Interest coverage",
    "Char4_FirmLeverageBeforeFiling": "Leverage",
}


### Load data ###

required_columns = [treatment, industry, *characteristics]
data = pd.read_csv(DATA_PATH, usecols=required_columns)

if set(data[treatment].dropna().unique()) != {0, 1}:
    raise ValueError("Treatment_Judge must contain only zero and one.")


### Continuous balance ###

balance_rows = []

for variable, label in characteristics.items():
    judge_zero = data.loc[data[treatment] == 0, variable].dropna().to_numpy()
    judge_one = data.loc[data[treatment] == 1, variable].dropna().to_numpy()

    n_zero = judge_zero.size
    n_one = judge_one.size
    mean_zero = judge_zero.mean()
    mean_one = judge_one.mean()
    variance_zero = judge_zero.var(ddof=1)
    variance_one = judge_one.var(ddof=1)
    # Differences are Judge 1 minus Judge 0.
    difference = mean_one - mean_zero
    # Welch allows unequal group variances.
    standard_error = np.sqrt(variance_zero / n_zero + variance_one / n_one)
    t_statistic = difference / standard_error
    welch_df = (variance_zero / n_zero + variance_one / n_one) ** 2 / (
        (variance_zero / n_zero) ** 2 / (n_zero - 1)
        + (variance_one / n_one) ** 2 / (n_one - 1)
    )
    # Use both tails of the Welch t distribution.
    p_value = 2 * stats.t.sf(abs(t_statistic), df=welch_df)
    # Standardize by the square root of the average group variance.
    pooled_scale = np.sqrt((variance_zero + variance_one) / 2)
    standardized_difference = difference / pooled_scale

    balance_rows.append(
        {
            "Characteristic": label,
            "Judge 0 N": n_zero,
            "Judge 0 mean": mean_zero,
            "Judge 1 N": n_one,
            "Judge 1 mean": mean_one,
            "Difference": difference,
            "Standard error": standard_error,
            "t statistic": t_statistic,
            "p value": p_value,
            "Standardized difference": standardized_difference,
        }
    )

balance = pd.DataFrame(balance_rows)
balance.to_csv(TMP_DIR / "Question_3_Continuous_Balance.csv", index=False)

balance_display = balance.copy()
for column in ["Judge 0 N", "Judge 1 N"]:
    balance_display[column] = balance_display[column].map(lambda value: f"{value:,}")
for column in [
    "Judge 0 mean",
    "Judge 1 mean",
    "Difference",
]:
    balance_display[column] = balance_display[column].map(lambda value: f"{value:.3f}")
balance_display["Standard error"] = balance_display["Standard error"].map(
    lambda value: "$<0.001$" if abs(value) < 0.001 else f"{value:.3f}"
)
balance_display["t statistic"] = balance_display["t statistic"].map(
    lambda value: f"{value:.3f}"
)
balance_display["p value"] = balance_display["p value"].map(
    lambda value: "$<0.001$" if value < 0.001 else f"{value:.3f}"
)
balance_display["Standardized difference"] = balance_display[
    "Standardized difference"
].map(lambda value: f"{value:.3f}")

balance_tabular = balance_display.to_latex(
    index=False,
    escape=False,
    column_format="lrrrrrrrrr",
)
balance_table = (
    "\\begin{table}[!htbp]\n"
    "\\centering\n"
    "\\caption{Continuous characteristic balance by judge}\n"
    "\\label{tab:q3_continuous_balance}\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    f"{balance_tabular}"
    "}\n"
    "\\begin{minipage}{0.96\\textwidth}\n"
    "\\footnotesize Notes: Difference equals the judge 1 mean minus the judge 0 mean. Standard errors and p values use the Welch two sample comparison. Values below 0.001 are displayed as $<0.001$.\n"
    "\\end{minipage}\n"
    "\\end{table}\n"
)
(OUTPUT_DIR / "Table_3_1_Continuous_Balance.tex").write_text(
    balance_table,
    encoding="utf-8",
)


### Industry balance ###

industry_counts = pd.crosstab(data[industry], data[treatment]).reindex(
    columns=[0, 1],
    fill_value=0,
)
industry_counts = industry_counts.sort_index()
judge_totals = data[treatment].value_counts().reindex([0, 1])

industry_rows = []
for industry_name, row in industry_counts.iterrows():
    # Shares use each judge's total cases.
    share_zero = row[0] / judge_totals[0]
    share_one = row[1] / judge_totals[1]
    industry_rows.append(
        {
            "Industry": industry_name,
            "Judge 0 count": int(row[0]),
            "Judge 0 share": share_zero,
            "Judge 1 count": int(row[1]),
            "Judge 1 share": share_one,
            "Difference in percentage points": 100 * (share_one - share_zero),
        }
    )

industry_balance = pd.DataFrame(industry_rows)
industry_balance.to_csv(TMP_DIR / "Question_3_Industry_Balance.csv", index=False)

industry_display = industry_balance.copy()
for column in ["Judge 0 count", "Judge 1 count"]:
    industry_display[column] = industry_display[column].map(lambda value: f"{value:,}")
for column in ["Judge 0 share", "Judge 1 share"]:
    industry_display[column] = industry_display[column].map(
        lambda value: f"{100 * value:.3f}\\%"
    )
industry_display["Difference in percentage points"] = industry_display[
    "Difference in percentage points"
].map(lambda value: f"{value:.3f}")

industry_tabular = industry_display.to_latex(
    index=False,
    escape=False,
    column_format="lrrrrr",
)
industry_table = (
    "\\begin{table}[!htbp]\n"
    "\\centering\n"
    "\\caption{Industry distributions by judge}\n"
    "\\label{tab:q3_industry_balance}\n"
    "\\resizebox{0.90\\textwidth}{!}{%\n"
    f"{industry_tabular}"
    "}\n"
    "\\begin{minipage}{0.96\\textwidth}\n"
    "\\scriptsize Notes: Each share uses the total number of cases assigned to that judge. Difference is measured in percentage points and equals judge 1 minus judge 0.\n"
    "\\end{minipage}\n"
    "\\end{table}\n"
)
(OUTPUT_DIR / "Table_3_2_Industry_Balance.tex").write_text(
    industry_table,
    encoding="utf-8",
)


### Distribution plots ###

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)

figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
colors = {0: "#4C78A8", 1: "#E45756"}

for axis, (variable, label) in zip(axes.flat, characteristics.items()):
    all_values = data[variable].dropna().to_numpy()
    grid = np.linspace(all_values.min(), all_values.max(), 1500)

    for judge in [0, 1]:
        values = np.sort(
            data.loc[data[treatment] == judge, variable].dropna().to_numpy()
        )
        # ECDF is the share at or below each grid value.
        empirical_cdf = np.searchsorted(values, grid, side="right") / values.size
        axis.plot(
            grid,
            empirical_cdf,
            color=colors[judge],
            linewidth=1.8,
            label=f"Judge {judge}",
        )

    axis.set_title(label)
    axis.set_xlabel(label)
    axis.set_ylabel("Empirical cumulative probability")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)

figure.suptitle("Prefiling characteristic distributions by judge", fontsize=14)
figure.savefig(
    OUTPUT_DIR / "Figure_3_1_Continuous_ECDF_By_Judge.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


### Density plots ###

figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)

for axis, (variable, label) in zip(axes.flat, characteristics.items()):
    all_values = data[variable].dropna().to_numpy()
    # Both groups use the same bin edges.
    bin_edges = np.linspace(all_values.min(), all_values.max(), 51)

    for judge in [0, 1]:
        values = data.loc[data[treatment] == judge, variable].dropna().to_numpy()
        # Each group's density integrates to one.
        density, _ = np.histogram(values, bins=bin_edges, density=True)
        axis.stairs(
            density,
            bin_edges,
            baseline=0,
            fill=True,
            color=colors[judge],
            alpha=0.25,
        )
        axis.stairs(
            density,
            bin_edges,
            color=colors[judge],
            linewidth=1.2,
            label=f"Judge {judge}",
        )

    axis.set_title(label)
    axis.set_xlabel(label)
    axis.set_ylabel("Probability density")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)

figure.suptitle("Prefiling characteristic densities by judge", fontsize=14)
figure.savefig(
    OUTPUT_DIR / "Figure_3_2_Continuous_Density_By_Judge.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


### Industry plot ###

industry_names = industry_balance["Industry"].tolist()
shares_zero = 100 * industry_balance["Judge 0 share"].to_numpy()
shares_one = 100 * industry_balance["Judge 1 share"].to_numpy()
x_positions = np.arange(len(industry_names))
bar_width = 0.36

figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
bars_zero = axis.bar(
    x_positions - bar_width / 2,
    shares_zero,
    width=bar_width,
    color=colors[0],
    label="Judge 0",
)
bars_one = axis.bar(
    x_positions + bar_width / 2,
    shares_one,
    width=bar_width,
    color=colors[1],
    label="Judge 1",
)

axis.bar_label(bars_zero, fmt="%.3f", padding=2, fontsize=8)
axis.bar_label(bars_one, fmt="%.3f", padding=2, fontsize=8)
axis.set_xticks(x_positions, industry_names)
axis.set_ylabel("Share of judge cases, percent")
axis.set_title("Industry distributions by judge")
axis.grid(axis="y", alpha=0.20)
axis.legend(frameon=False)
axis.set_ylim(0, max(shares_zero.max(), shares_one.max()) * 1.15)

figure.savefig(
    OUTPUT_DIR / "Figure_3_3_Industry_Shares_By_Judge.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)
