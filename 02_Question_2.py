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
import pandas as pd

# Plot numeric distributions in source units; summarize categories in Question 1.
plot_specs = [
    (
        "FirmNumber",
        "Firm number",
        "Firm number",
        "Figure_2_1_FirmNumber_Distribution.png",
    ),
    (
        "Outcome_FirmValue1YearAfterFiling",
        "Firm value after filing",
        "Firm value one year after filing, millions of dollars",
        "Figure_2_2_Outcome_Value_Distribution.png",
    ),
    (
        "Char1_FirmValueBeforeFiling",
        "Firm value before filing",
        "Firm value before filing, millions of dollars",
        "Figure_2_3_Prefiling_Value_Distribution.png",
    ),
    (
        "Char2_FirmProfitMarginBeforeFiling",
        "Profit margin",
        "Profit margin before filing, decimal ratio",
        "Figure_2_4_Profit_Margin_Distribution.png",
    ),
    (
        "Char3_FirmInterestCoverageBeforeFiling",
        "Interest coverage",
        "Interest coverage before filing",
        "Figure_2_5_Interest_Coverage_Distribution.png",
    ),
    (
        "Char4_FirmLeverageBeforeFiling",
        "Leverage",
        "Leverage before filing, decimal ratio",
        "Figure_2_6_Leverage_Distribution.png",
    ),
]


### Load data ###

plot_columns = [spec[0] for spec in plot_specs]
# Read the full sample without trimming or sampling.
data = pd.read_csv(DATA_PATH, usecols=plot_columns)


### Distribution plots ###

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)

for variable, title, x_label, filename in plot_specs:
    # Remove missing values separately for each variable.
    values = data[variable].dropna()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.6),
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.14, hspace=0.08)

    # Show observation counts across 50 equal width bins.
    axes[0].hist(
        values,
        bins=50,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.35,
    )
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(title)
    axes[0].grid(axis="y", alpha=0.20)
    axes[0].tick_params(axis="x", labelbottom=False)

    # Show the median, quartiles, and points beyond 1.5 IQR whiskers.
    axes[1].boxplot(
        values,
        vert=False,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#A6CEE3", "edgecolor": "#2F4B7C"},
        medianprops={"color": "#E45756", "linewidth": 1.5},
        whiskerprops={"color": "#2F4B7C"},
        capprops={"color": "#2F4B7C"},
        flierprops={
            "marker": ".",
            "markersize": 1.5,
            "markerfacecolor": "#7F7F7F",
            "markeredgecolor": "none",
            "alpha": 0.30,
        },
    )
    axes[1].set_xlabel(x_label)
    axes[1].set_yticks([])
    axes[1].grid(axis="x", alpha=0.20)

    figure.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(figure)
