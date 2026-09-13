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

import pandas as pd


### Load data ###

data = pd.read_csv(DATA_PATH)

# Industry is categorical; the identifier and judge dummy remain numeric.
numeric_columns = [
    "FirmNumber",
    "Outcome_FirmValue1YearAfterFiling",
    "Treatment_Judge",
    "Char1_FirmValueBeforeFiling",
    "Char2_FirmProfitMarginBeforeFiling",
    "Char3_FirmInterestCoverageBeforeFiling",
    "Char4_FirmLeverageBeforeFiling",
]


### Summary statistics ###

# Use source units and the sample standard deviation.
summary = (
    data[numeric_columns]
    .describe(percentiles=[0.25, 0.50, 0.75])
    .T[["mean", "50%", "min", "max", "25%", "75%", "std", "count"]]
)
summary.columns = [
    "Mean",
    "Median",
    "Minimum",
    "Maximum",
    "25th percentile",
    "75th percentile",
    "Standard deviation",
    "Nonmissing count",
]
summary.index.name = "Variable"
summary.to_csv(TMP_DIR / "Question_1_Summary_Statistics.csv")

# Round displayed results only; keep counts as integers.
summary_display = summary.copy()
variable_labels = {
    "FirmNumber": "Firm number",
    "Outcome_FirmValue1YearAfterFiling": "Firm value one year after filing",
    "Treatment_Judge": "Judge 1 indicator",
    "Char1_FirmValueBeforeFiling": "Firm value before filing",
    "Char2_FirmProfitMarginBeforeFiling": "Profit margin before filing",
    "Char3_FirmInterestCoverageBeforeFiling": "Interest coverage before filing",
    "Char4_FirmLeverageBeforeFiling": "Leverage before filing",
}
summary_display.index = summary_display.index.map(variable_labels)
for column in summary_display.columns[:-1]:
    summary_display[column] = summary_display[column].map(lambda value: f"{value:,.3f}")
summary_display["Nonmissing count"] = summary_display["Nonmissing count"].map(
    lambda value: f"{int(value):,}"
)
summary_display = summary_display.reset_index(names="Variable")

summary_tabular = summary_display.to_latex(
    index=False,
    escape=True,
    column_format="lrrrrrrrr",
)
summary_table = (
    "\\begin{table}[!htbp]\n"
    "\\centering\n"
    "\\caption{Summary statistics}\n"
    "\\label{tab:q1_summary}\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    f"{summary_tabular}"
    "}\n"
    "\\begin{minipage}{0.96\\textwidth}\n"
    "\\footnotesize Notes: Firm values are measured in millions of dollars. Profit margin and leverage are decimal ratios. Firm number is an identifier.\n"
    "\\end{minipage}\n"
    "\\end{table}\n"
)
(OUTPUT_DIR / "Table_1_1_Summary_Statistics.tex").write_text(
    summary_table,
    encoding="utf-8",
)


### Category shares ###

share_rows = []

# Include missing categories and use the full sample as the denominator.
judge_counts = data["Treatment_Judge"].value_counts(dropna=False).sort_index()
for category, count in judge_counts.items():
    category_label = "Missing" if pd.isna(category) else f"Judge {int(category)}"
    share_rows.append(
        {
            "Variable": "Treatment Judge",
            "Group": category_label,
            "Count": int(count),
            "Share": count / len(data),
        }
    )

industry_counts = data["Char5_FirmIndustry"].value_counts(dropna=False).sort_index()
for category, count in industry_counts.items():
    category_label = "Missing" if pd.isna(category) else str(category)
    share_rows.append(
        {
            "Variable": "Firm Industry",
            "Group": category_label,
            "Count": int(count),
            "Share": count / len(data),
        }
    )

shares = pd.DataFrame(share_rows)
shares.to_csv(TMP_DIR / "Question_1_Category_Shares.csv", index=False)

# Convert decimal shares to percentages for the table.
shares_display = shares.copy()
shares_display["Count"] = shares_display["Count"].map(lambda value: f"{value:,}")
shares_display["Share"] = shares_display["Share"].map(lambda value: f"{100 * value:.3f}\\%")

shares_tabular = shares_display.to_latex(
    index=False,
    escape=False,
    column_format="llrr",
)
shares_table = (
    "\\begin{table}[!htbp]\n"
    "\\centering\n"
    "\\caption{Judge and industry shares}\n"
    "\\label{tab:q1_shares}\n"
    f"{shares_tabular}"
    "\\end{table}\n"
)
(OUTPUT_DIR / "Table_1_2_Categorical_Shares.tex").write_text(
    shares_table,
    encoding="utf-8",
)
