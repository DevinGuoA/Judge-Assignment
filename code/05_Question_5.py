from pathlib import Path
import json
import os


### Setup ###

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "PhdFinance_FNCE7020_HW1_Data.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
RESULTS_DIR = PROJECT_DIR / "tmp" / "results"
CACHE_DIR = PROJECT_DIR / "tmp" / "cache"
MPL_DIR = PROJECT_DIR / "tmp" / "matplotlib"

for directory in [OUTPUT_DIR, RESULTS_DIR, CACHE_DIR, MPL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import numpy as np
import pandas as pd
from scipy import stats


### Data ###

columns = [
    "Outcome_FirmValue1YearAfterFiling",
    "Treatment_Judge",
    "Char1_FirmValueBeforeFiling",
    "Char5_FirmIndustry",
]

data = pd.read_csv(
    DATA_PATH,
    usecols=columns,
    dtype={
        "Outcome_FirmValue1YearAfterFiling": "float64",
        "Treatment_Judge": "int8",
        "Char1_FirmValueBeforeFiling": "float64",
        "Char5_FirmIndustry": "category",
    },
)

if data[columns].isna().any().any():
    raise ValueError("Required variables contain missing values.")

judge_values = set(data["Treatment_Judge"].unique().tolist())
if judge_values != {0, 1}:
    raise ValueError("Treatment_Judge must contain only 0 and 1.")

before_all = data["Char1_FirmValueBeforeFiling"].to_numpy()
after_all = data["Outcome_FirmValue1YearAfterFiling"].to_numpy()
judge_all = data["Treatment_Judge"].to_numpy()
tech_all = (
    data["Char5_FirmIndustry"].astype("string") == "Tech"
).to_numpy(dtype=bool)

zero_before = before_all == 0
excluded_zero_count = int(zero_before.sum())
if excluded_zero_count != 1:
    raise ValueError("Expected exactly one zero pre filing value.")

# Zero pre filing value makes percent change undefined.
valid_outcome = ~zero_before
before = before_all[valid_outcome]
after = after_all[valid_outcome]
judge1 = judge_all[valid_outcome]
tech = tech_all[valid_outcome]
percent_change = 100 * (after - before) / before


### Assignment evidence ###

# Assignment comparisons retain the full sample.
judge0_all = judge_all == 0
judge1_all = judge_all == 1

assignment_rows = []
assignment_comparisons = [
    (
        "Pre filing firm value, millions",
        before_all[judge0_all],
        before_all[judge1_all],
        1.0,
    ),
    (
        "Tech share, percent",
        tech_all[judge0_all].astype(float),
        tech_all[judge1_all].astype(float),
        100.0,
    ),
]

for label, judge0_values, judge1_values, scale in assignment_comparisons:
    # Tech shares are scaled to percent.
    judge0_values = scale * judge0_values
    judge1_values = scale * judge1_values
    n_judge0 = judge0_values.size
    n_judge1 = judge1_values.size
    mean_judge0 = float(judge0_values.mean())
    mean_judge1 = float(judge1_values.mean())
    variance_judge0 = float(judge0_values.var(ddof=1))
    variance_judge1 = float(judge1_values.var(ddof=1))
    # Assignment differences are Judge 1 minus Judge 0.
    difference = mean_judge1 - mean_judge0
    # Welch allows unequal group variances.
    standard_error = np.sqrt(
        variance_judge0 / n_judge0 + variance_judge1 / n_judge1
    )
    welch_df = standard_error**4 / (
        (variance_judge0 / n_judge0) ** 2 / (n_judge0 - 1)
        + (variance_judge1 / n_judge1) ** 2 / (n_judge1 - 1)
    )
    t_statistic = difference / standard_error
    # Use both tails of the Welch t distribution.
    p_value = 2 * stats.t.sf(abs(t_statistic), df=welch_df)

    assignment_rows.append(
        {
            "Characteristic": label,
            "Judge 0 N": n_judge0,
            "Judge 0 mean": mean_judge0,
            "Judge 1 N": n_judge1,
            "Judge 1 mean": mean_judge1,
            "Difference": difference,
            "Standard error": standard_error,
            "t statistic": t_statistic,
            "p value": p_value,
        }
    )

assignment_results = pd.DataFrame(assignment_rows)


### Outcome evidence ###

# The cutoff is the median among firms with valid outcomes.
median_value = float(np.median(before))
# High value strictly exceeds the median.
high_value = before > median_value
outcome_rows = []
outcome_comparisons = [
    (
        "Pre filing value",
        "At or below median",
        percent_change[~high_value],
        "Above median",
        percent_change[high_value],
    ),
    (
        "Tech status",
        "Non Tech",
        percent_change[~tech],
        "Tech",
        percent_change[tech],
    ),
]

for comparison, reference_label, reference_values, comparison_label, comparison_values in outcome_comparisons:
    n_reference = reference_values.size
    n_comparison = comparison_values.size
    mean_reference = float(reference_values.mean())
    mean_comparison = float(comparison_values.mean())
    variance_reference = float(reference_values.var(ddof=1))
    variance_comparison = float(comparison_values.var(ddof=1))
    # Differences: above minus at/below median; Tech minus non Tech.
    difference = mean_comparison - mean_reference
    standard_error = np.sqrt(
        variance_reference / n_reference
        + variance_comparison / n_comparison
    )
    welch_df = standard_error**4 / (
        (variance_reference / n_reference) ** 2 / (n_reference - 1)
        + (variance_comparison / n_comparison) ** 2 / (n_comparison - 1)
    )
    t_statistic = difference / standard_error
    p_value = 2 * stats.t.sf(abs(t_statistic), df=welch_df)

    outcome_rows.append(
        {
            "Comparison": comparison,
            "Reference label": reference_label,
            "Reference N": n_reference,
            "Reference mean": mean_reference,
            "Comparison label": comparison_label,
            "Comparison N": n_comparison,
            "Comparison mean": mean_comparison,
            "Difference": difference,
            "Standard error": standard_error,
            "t statistic": t_statistic,
            "p value": p_value,
        }
    )

outcome_results = pd.DataFrame(outcome_rows)

value_assignment = assignment_rows[0]
tech_assignment = assignment_rows[1]
value_outcome = outcome_rows[0]
tech_outcome = outcome_rows[1]

selection_consistent = bool(
    value_assignment["Difference"] > 0 and tech_assignment["Difference"] > 0
)
outcome_pattern_consistent = bool(
    value_outcome["Difference"] > 0 and tech_outcome["Difference"] > 0
)

# Descriptive signs support the expected bias direction.
if selection_consistent and outcome_pattern_consistent:
    bias_direction = "upward"
elif not selection_consistent:
    bias_direction = "ambiguous from the observed selection evidence"
else:
    bias_direction = "ambiguous from the observed outcome evidence"


### Assignment table ###

assignment_display = assignment_results.copy()
for column in ["Judge 0 N", "Judge 1 N"]:
    assignment_display[column] = assignment_display[column].map(
        lambda value: f"{value:,}"
    )
for column in ["Judge 0 mean", "Judge 1 mean", "Difference"]:
    assignment_display[column] = assignment_display[column].map(
        lambda value: f"{value:.3f}"
    )
assignment_display["Standard error"] = assignment_display["Standard error"].map(
    lambda value: f"{value:.3f}"
)
assignment_display["t statistic"] = assignment_display["t statistic"].map(
    lambda value: f"{value:.3f}"
)
assignment_display["p value"] = assignment_display["p value"].map(
    lambda value: "$<0.001$" if value < 0.001 else f"{value:.3f}"
)

assignment_tabular = assignment_display.to_latex(
    index=False,
    escape=False,
    column_format="lrrrrrrrr",
)
assignment_table = (
    "\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Judge assignment and pre filing characteristics}\n"
    "\\label{tab:q5_assignment}\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    f"{assignment_tabular}"
    "}\n"
    "\\begin{minipage}{0.96\\textwidth}\n"
    "\\footnotesize Notes: Difference equals Judge 1 minus Judge 0. Firm value is measured in millions of dollars. Tech means and differences are measured in percent and percentage points. Standard errors and p values use two sided Welch comparisons calculated from unrounded values. P values below 0.001 are displayed as $<0.001$.\n"
    "\\end{minipage}\n"
    "\\end{table}\n"
)
(OUTPUT_DIR / "Table_5_1_Assignment_Evidence.tex").write_text(
    assignment_table,
    encoding="utf-8",
)


### Outcome table ###

value_row = outcome_rows[0]
tech_row = outcome_rows[1]

outcome_table_lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Outcome differences by pre filing value and Tech status}",
    r"\label{tab:q5_outcomes}",
    r"\resizebox{0.98\textwidth}{!}{%",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"\multicolumn{7}{l}{\textit{Panel A: Pre filing value}} \\",
    r"\addlinespace",
    r"Measure & At or below median & Above median & Difference & Standard error & t statistic & p value \\",
    r"\midrule",
    f"Mean percent change & {value_row['Reference mean']:.3f} & {value_row['Comparison mean']:.3f} & {value_row['Difference']:.3f} & {value_row['Standard error']:.3f} & {value_row['t statistic']:.3f} & " + (r"$<0.001$" if value_row["p value"] < 0.001 else f"{value_row['p value']:.3f}") + r" \\",
    f"Observations & {value_row['Reference N']:,} & {value_row['Comparison N']:,} & & & & " + r"\\",
    r"\midrule",
    r"\multicolumn{7}{l}{\textit{Panel B: Tech status}} \\",
    r"\addlinespace",
    r"Measure & Non Tech & Tech & Difference & Standard error & t statistic & p value \\",
    r"\midrule",
    f"Mean percent change & {tech_row['Reference mean']:.3f} & {tech_row['Comparison mean']:.3f} & {tech_row['Difference']:.3f} & {tech_row['Standard error']:.3f} & {tech_row['t statistic']:.3f} & " + (r"$<0.001$" if tech_row["p value"] < 0.001 else f"{tech_row['p value']:.3f}") + r" \\",
    f"Observations & {tech_row['Reference N']:,} & {tech_row['Comparison N']:,} & & & & " + r"\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\par\smallskip",
    r"\begin{minipage}{0.94\linewidth}",
    f"\\footnotesize Notes: High value means above the sample median pre filing value of {median_value:.3f} million dollars. Difference equals above median minus at or below median in Panel A and Tech minus non Tech in Panel B. Means, differences, and standard errors are measured in percentage points. Standard errors and p values use two sided Welch comparisons calculated from unrounded values. One observation with a zero denominator is excluded. P values below 0.001 are displayed as $<0.001$.",
    r"\end{minipage}",
    r"\end{table}",
    "",
]
(OUTPUT_DIR / "Table_5_2_Outcome_Associations.tex").write_text(
    "\n".join(outcome_table_lines), encoding="utf-8"
)

results = {
    "question": 5,
    "outcome_definition": "100 * (after - before) / before",
    "excluded_zero_denominator_count": excluded_zero_count,
    "assignment_n_obs": int(len(data)),
    "outcome_n_obs": int(valid_outcome.sum()),
    "selection_evidence": {
        "mean_pre_filing_value_judge0": value_assignment["Judge 0 mean"],
        "mean_pre_filing_value_judge1": value_assignment["Judge 1 mean"],
        "judge1_minus_judge0_value": value_assignment["Difference"],
        "value_difference_standard_error": value_assignment["Standard error"],
        "value_difference_t_statistic": value_assignment["t statistic"],
        "value_difference_p_value": value_assignment["p value"],
        "tech_share_percent_judge0": tech_assignment["Judge 0 mean"],
        "tech_share_percent_judge1": tech_assignment["Judge 1 mean"],
        "judge1_minus_judge0_tech_share_percentage_points": float(
            tech_assignment["Difference"]
        ),
        "tech_share_difference_standard_error": tech_assignment["Standard error"],
        "tech_share_difference_t_statistic": tech_assignment["t statistic"],
        "tech_share_difference_p_value": tech_assignment["p value"],
    },
    "outcome_evidence": {
        "median_pre_filing_value": median_value,
        "mean_percent_change_lower_value": value_outcome["Reference mean"],
        "mean_percent_change_high_value": value_outcome["Comparison mean"],
        "high_minus_lower_value_percentage_points": float(
            value_outcome["Difference"]
        ),
        "high_value_difference_standard_error": value_outcome["Standard error"],
        "high_value_difference_t_statistic": value_outcome["t statistic"],
        "high_value_difference_p_value": value_outcome["p value"],
        "mean_percent_change_nontech": tech_outcome["Reference mean"],
        "mean_percent_change_tech": tech_outcome["Comparison mean"],
        "tech_minus_nontech_percentage_points": float(tech_outcome["Difference"]),
        "tech_difference_standard_error": tech_outcome["Standard error"],
        "tech_difference_t_statistic": tech_outcome["t statistic"],
        "tech_difference_p_value": tech_outcome["p value"],
    },
    "selection_consistent": selection_consistent,
    "outcome_pattern_consistent": outcome_pattern_consistent,
    "omitted_variable_bias_direction": bias_direction,
}
(RESULTS_DIR / "Question_5_results.json").write_text(
    json.dumps(results, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"Question 5 complete with {valid_outcome.sum():,} outcome observations.")
print(f"Expected omitted variable bias: {bias_direction}.")
