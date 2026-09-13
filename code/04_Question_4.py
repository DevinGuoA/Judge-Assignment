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
]

data = pd.read_csv(
    DATA_PATH,
    usecols=columns,
    dtype={
        "Outcome_FirmValue1YearAfterFiling": "float64",
        "Treatment_Judge": "int8",
        "Char1_FirmValueBeforeFiling": "float64",
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
# The dependent variable is in percent, not decimal units.
percent_change = 100 * (after - before) / before


### OLS ###

# X columns and coefficient order: constant, Judge 1.
x = np.column_stack((np.ones(len(data)), judge1))
beta = np.linalg.lstsq(x, percent_change, rcond=None)[0]
residual = percent_change - beta[0] - beta[1] * judge1
n_obs, n_parameters = x.shape
degrees_freedom = n_obs - n_parameters
residual_sum_squares = float(
    np.sum(np.square(residual), dtype=np.float64)
)
residual_variance = residual_sum_squares / degrees_freedom
# Conventional homoskedastic covariance: s^2 * (X'X)^(-1).
covariance = residual_variance * np.linalg.inv(x.T @ x)
standard_errors = np.sqrt(np.diag(covariance))
t_statistics = beta / standard_errors
# Two sided tests of each coefficient against zero.
p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_freedom)

total_sum_squares = float(
    np.sum(
        np.square(percent_change - percent_change.mean()),
        dtype=np.float64,
    )
)
r_squared = 1 - residual_sum_squares / total_sum_squares

# The Judge 1 coefficient equals mean(Judge 1) minus mean(Judge 0).
mean_judge0 = float(percent_change[judge1 == 0].mean())
mean_judge1 = float(percent_change[judge1 == 1].mean())
confidence_low = float(beta[1] - stats.t.ppf(0.975, degrees_freedom) * standard_errors[1])
confidence_high = float(beta[1] + stats.t.ppf(0.975, degrees_freedom) * standard_errors[1])


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
    r"\caption{Simple OLS regression of percent change in firm value}",
    r"\label{tab:q4_ols}",
    r"\begin{tabular}{lc}",
    r"\toprule",
    r" & (1) \\",
    r" & Percent change in firm value \\",
    r"\midrule",
    f"Judge 1 & {beta[1]:.3f}{stars[1]} " + r"\\",
    f" & ({standard_errors[1]:.3f}) " + r"\\",
    r"\addlinespace[2pt]",
    f"Constant & {beta[0]:.3f}{stars[0]} " + r"\\",
    f" & ({standard_errors[0]:.3f}) " + r"\\",
    r"\midrule",
    f"Observations & {n_obs:,} " + r"\\",
    f"$R^2$ & {r_squared:.3f} " + r"\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip",
    r"\begin{minipage}{0.76\linewidth}",
    (
        r"\footnotesize Notes: The regression specification is $Y_i = \alpha + \beta\,\mathrm{Judge\,1}_i + \varepsilon_i$. The dependent variable is $100(V_{after}-V_{before})/V_{before}$. Judge 0 is the omitted category. Conventional homoskedastic standard errors are in parentheses. "
        f"The $t$ statistics are {t_statistics[1]:.3f} for Judge 1 and {t_statistics[0]:.3f} for the constant. Both $p$ values are below 0.001. "
        r"Significance levels use two sided tests. One observation with a zero denominator is excluded. $^{*}p<0.10$, $^{**}p<0.05$, and $^{***}p<0.01$."
    ),
    r"\end{minipage}",
    r"\end{table}",
    "",
]
(OUTPUT_DIR / "Table_4_OLS_Judge.tex").write_text(
    "\n".join(table_lines), encoding="utf-8"
)

# Preserve full precision in the saved numerical results.
results = {
    "question": 4,
    "outcome_definition": "100 * (after - before) / before",
    "excluded_zero_denominator_count": excluded_zero_count,
    "n_obs": int(n_obs),
    "degrees_freedom": int(degrees_freedom),
    "coefficients": {
        "constant": float(beta[0]),
        "judge1": float(beta[1]),
    },
    "standard_errors": {
        "constant": float(standard_errors[0]),
        "judge1": float(standard_errors[1]),
    },
    "t_statistics": {
        "constant": float(t_statistics[0]),
        "judge1": float(t_statistics[1]),
    },
    "p_values": {
        "constant": float(p_values[0]),
        "judge1": float(p_values[1]),
    },
    "judge1_confidence_interval_95": [confidence_low, confidence_high],
    "mean_percent_change": {
        "judge0": mean_judge0,
        "judge1": mean_judge1,
    },
    "r_squared": float(r_squared),
}
(RESULTS_DIR / "Question_4_results.json").write_text(
    json.dumps(results, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"Question 4 complete with {n_obs:,} observations.")
print(f"Judge 1 coefficient: {beta[1]:.3f}")
