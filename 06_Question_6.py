from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats


### Setup ###

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "PhdFinance_FNCE7020_HW1_Data.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
RESULTS_DIR = PROJECT_DIR / "tmp" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
# Zero pre filing value makes percentage change undefined.
zero_before = before_all == 0
excluded_zero_count = int(zero_before.sum())
if excluded_zero_count != 1:
    raise ValueError("Expected exactly one zero pre filing value.")

data = data.loc[~zero_before].copy()
# Firm values are in millions of dollars.
before = data["Char1_FirmValueBeforeFiling"].to_numpy()
after = data["Outcome_FirmValue1YearAfterFiling"].to_numpy()
# Judge 1 equals one; Judge 0 is the omitted category.
judge1 = data["Treatment_Judge"].to_numpy(dtype="float64")
# Tech equals one; all other industries form the omitted category.
tech = (
    data["Char5_FirmIndustry"].astype("string").eq("Tech").to_numpy(dtype="float64")
)
# The dependent variable is in percent, not decimal units.
percent_change = 100 * (after - before) / before


### Adjusted OLS ###

# X columns and coefficient order: constant, Judge 1, Tech, pre filing value.
x_adjusted = np.column_stack((np.ones(len(data)), judge1, tech, before))
beta_adjusted = np.linalg.lstsq(x_adjusted, percent_change, rcond=None)[0]
residual_adjusted = (
    percent_change
    - beta_adjusted[0]
    - beta_adjusted[1] * judge1
    - beta_adjusted[2] * tech
    - beta_adjusted[3] * before
)
n_obs, n_parameters = x_adjusted.shape
degrees_freedom = n_obs - n_parameters
adjusted_residual_sum_squares = float(
    np.sum(np.square(residual_adjusted), dtype=np.float64)
)
residual_variance = adjusted_residual_sum_squares / degrees_freedom
# Conventional homoskedastic covariance: s^2 * (X'X)^(-1).
covariance = residual_variance * np.linalg.inv(x_adjusted.T @ x_adjusted)
standard_errors = np.sqrt(np.diag(covariance))
t_statistics = beta_adjusted / standard_errors
# Two sided tests of each coefficient against zero.
p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_freedom)

total_sum_squares = float(
    np.sum(
        np.square(percent_change - percent_change.mean()),
        dtype=np.float64,
    )
)
r_squared = 1 - adjusted_residual_sum_squares / total_sum_squares


### Question 4 Comparison ###

# Reestimate Question 4 on the same outcome sample.
x_simple = np.column_stack((np.ones(len(data)), judge1))
beta_simple = np.linalg.lstsq(x_simple, percent_change, rcond=None)[0]
residual_simple = percent_change - beta_simple[0] - beta_simple[1] * judge1
simple_degrees_freedom = n_obs - x_simple.shape[1]
simple_residual_variance = float(
    np.sum(np.square(residual_simple), dtype=np.float64)
    / simple_degrees_freedom
)
simple_covariance = simple_residual_variance * np.linalg.inv(x_simple.T @ x_simple)
simple_standard_errors = np.sqrt(np.diag(simple_covariance))
simple_t_statistics = beta_simple / simple_standard_errors
simple_p_values = 2 * stats.t.sf(
    np.abs(simple_t_statistics), simple_degrees_freedom
)

# Signed difference: adjusted minus simple, in percentage points.
judge_coefficient_change = float(beta_adjusted[1] - beta_simple[1])
judge_coefficient_percent_change = float(
    100 * judge_coefficient_change / abs(beta_simple[1])
)


### Outputs ###

# Assign stars before rounding estimates for display.
stars = [
    "$^{***}$"
    if value < 0.01
    else "$^{**}$"
    if value < 0.05
    else "$^{*}$"
    if value < 0.10
    else ""
    for value in p_values
]

table_lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{OLS regression of percent change in firm value with controls}",
    r"\label{tab:q6_ols}",
    r"\begin{tabular}{lc}",
    r"\toprule",
    r" & (1) \\",
    r" & Percent change in firm value \\",
    r"\midrule",
    f"Judge 1 & {beta_adjusted[1]:.3f}{stars[1]} " + r"\\",
    f" & ({standard_errors[1]:.3f}) " + r"\\",
    r"\addlinespace[2pt]",
    f"Tech & {beta_adjusted[2]:.3f}{stars[2]} " + r"\\",
    f" & ({standard_errors[2]:.3f}) " + r"\\",
    r"\addlinespace[2pt]",
    f"Pre filing firm value & {beta_adjusted[3]:.3f}{stars[3]} " + r"\\",
    f" & ({standard_errors[3]:.3f}) " + r"\\",
    r"\addlinespace[2pt]",
    f"Constant & {beta_adjusted[0]:.3f}{stars[0]} " + r"\\",
    f" & ({standard_errors[0]:.3f}) " + r"\\",
    r"\midrule",
    f"Observations & {n_obs:,} " + r"\\",
    f"$R^2$ & {r_squared:.3f} " + r"\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\begin{minipage}{0.90\linewidth}",
    (
        r"\footnotesize Notes: The regression specification is $Y_i = \alpha + \beta\,\mathrm{Judge\,1}_i + \gamma\,\mathrm{Tech}_i + \delta V_{before,i} + \varepsilon_i$. The dependent variable is $100(V_{after}-V_{before})/V_{before}$. Judge 0 and non Tech are the omitted categories. Tech equals one for firms in the Tech industry. Pre filing firm value is measured in millions of dollars. Conventional homoskedastic standard errors are in parentheses. "
        f"The $t$ statistics are {t_statistics[1]:.3f} for Judge 1, {t_statistics[2]:.3f} for Tech, {t_statistics[3]:.3f} for pre filing firm value, and {t_statistics[0]:.3f} for the constant. All $p$ values are below 0.001. "
        r"Significance levels use two sided tests. One observation with a zero denominator is excluded. $^{*}p<0.10$, $^{**}p<0.05$, and $^{***}p<0.01$."
    ),
    r"\end{minipage}",
    r"\end{table}",
    "",
]
(OUTPUT_DIR / "Table_6_Adjusted_OLS.tex").write_text(
    "\n".join(table_lines), encoding="utf-8"
)

# Preserve full precision in the saved numerical results.
results = {
    "question": 6,
    "outcome_definition": "100 * (after - before) / before",
    "excluded_zero_denominator_count": excluded_zero_count,
    "n_obs": int(n_obs),
    "degrees_freedom": int(degrees_freedom),
    "adjusted_model": {
        "variable_order": [
            "constant",
            "judge1",
            "tech",
            "pre_filing_value",
        ],
        "coefficients": [float(value) for value in beta_adjusted],
        "standard_errors": [float(value) for value in standard_errors],
        "t_statistics": [float(value) for value in t_statistics],
        "p_values": [float(value) for value in p_values],
        "r_squared": float(r_squared),
    },
    "question_4_comparison": {
        "judge1_coefficient": float(beta_simple[1]),
        "judge1_standard_error": float(simple_standard_errors[1]),
        "judge1_t_statistic": float(simple_t_statistics[1]),
        "judge1_p_value": float(simple_p_values[1]),
        "adjusted_minus_simple_percentage_points": judge_coefficient_change,
        "change_relative_to_absolute_simple_percent": judge_coefficient_percent_change,
    },
}
(RESULTS_DIR / "Question_6_results.json").write_text(
    json.dumps(results, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"Question 6 complete with {n_obs:,} observations.")
print(f"Adjusted Judge 1 coefficient: {beta_adjusted[1]:.3f}")
print(f"Simple Judge 1 coefficient: {beta_simple[1]:.3f}")
