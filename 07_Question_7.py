from pathlib import Path
import json

import numpy as np
import pandas as pd


### Setup ###

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "PhdFinance_FNCE7020_HW1_Data.csv"
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = ROOT / "tmp" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


### Load Data ###

data = pd.read_csv(DATA_PATH)
prefiling_value = data["Char1_FirmValueBeforeFiling"].to_numpy(dtype=float)
postfiling_value = data["Outcome_FirmValue1YearAfterFiling"].to_numpy(dtype=float)

percent_change = np.full(len(data), np.nan, dtype=float)
np.divide(
    postfiling_value - prefiling_value,
    prefiling_value,
    out=percent_change,
    where=prefiling_value != 0,
)
percent_change *= 100.0
data["PercentChange"] = percent_change

# Exclude the undefined percentage change from zero pre filing value.
valid_change = np.isfinite(data["PercentChange"].to_numpy())
undefined_change_count = int((~valid_change).sum())
analysis = data.loc[valid_change].copy()

if undefined_change_count != 1:
    raise ValueError("Expected exactly one undefined percent change observation.")


### Match Firms ###

match_parts = []
quality_rows = []

# Match within industry; controls remain available for reuse.
for industry in sorted(analysis["Char5_FirmIndustry"].unique()):
    industry_data = analysis.loc[analysis["Char5_FirmIndustry"].eq(industry)]
    controls = industry_data.loc[industry_data["Treatment_Judge"].eq(0)].copy()
    treated = industry_data.loc[industry_data["Treatment_Judge"].eq(1)].copy()

    # Sort by value; keep the smallest firm ID for duplicate values.
    controls = controls.sort_values(
        ["Char1_FirmValueBeforeFiling", "FirmNumber"], kind="mergesort"
    )
    controls = controls.drop_duplicates(
        subset="Char1_FirmValueBeforeFiling", keep="first"
    ).reset_index(drop=True)
    treated = treated.sort_values("FirmNumber", kind="mergesort").reset_index(drop=True)

    if controls.empty or treated.empty:
        raise ValueError(f"Both judge groups are required in {industry}.")

    control_values = controls["Char1_FirmValueBeforeFiling"].to_numpy()
    treated_values = treated["Char1_FirmValueBeforeFiling"].to_numpy()

    # Locate each insertion point and its two adjacent control values.
    insertion = np.searchsorted(control_values, treated_values, side="left")
    # Clip indices to the nearest boundary for values outside the range.
    lower_index = np.clip(insertion - 1, 0, len(control_values) - 1)
    upper_index = np.clip(insertion, 0, len(control_values) - 1)

    lower_distance = np.abs(treated_values - control_values[lower_index])
    upper_distance = np.abs(control_values[upper_index] - treated_values)
    # Equal distances select the lower control value.
    chosen_index = np.where(lower_distance <= upper_distance, lower_index, upper_index)

    matched_controls = controls.iloc[chosen_index].reset_index(drop=True)
    matched_values = matched_controls["Char1_FirmValueBeforeFiling"].to_numpy()
    absolute_distance = np.abs(treated_values - matched_values)
    # Flag values outside the observed industry control range; retain matches.
    outside_support = (treated_values < control_values[0]) | (
        treated_values > control_values[-1]
    )

    industry_matches = pd.DataFrame(
        {
            "Industry": industry,
            "TreatedFirmNumber": treated["FirmNumber"].to_numpy(),
            "ControlFirmNumber": matched_controls["FirmNumber"].to_numpy(),
            "TreatedPrefilingValue": treated_values,
            "ControlPrefilingValue": matched_values,
            "TreatedPercentChange": treated["PercentChange"].to_numpy(),
            "ControlPercentChange": matched_controls["PercentChange"].to_numpy(),
            "AbsoluteDistance": absolute_distance,
            "OutsideSupport": outside_support,
        }
    )
    match_parts.append(industry_matches)

    quality_rows.append(
        {
            "industry": industry,
            "treated_firms": int(len(industry_matches)),
            "unique_controls_used": int(
                industry_matches["ControlFirmNumber"].nunique()
            ),
            "mean_distance": float(industry_matches["AbsoluteDistance"].mean()),
            "median_distance": float(industry_matches["AbsoluteDistance"].median()),
            "p95_distance": float(
                industry_matches["AbsoluteDistance"].quantile(0.95)
            ),
            "maximum_distance": float(industry_matches["AbsoluteDistance"].max()),
            "outside_support": int(industry_matches["OutsideSupport"].sum()),
        }
    )

matches = pd.concat(match_parts, ignore_index=True)

expected_treated = int(analysis["Treatment_Judge"].eq(1).sum())
if len(matches) != expected_treated:
    raise ValueError("Every judge 1 firm must receive exactly one match.")


### Estimate Effects ###

# Control outcomes enter once per match, giving weights based on reuse.
treated_mean = float(matches["TreatedPercentChange"].mean())
matched_control_mean = float(matches["ControlPercentChange"].mean())
# ATT is the mean treated minus control outcome across matched pairs.
att = float(
    (matches["TreatedPercentChange"] - matches["ControlPercentChange"]).mean()
)

# Question 6 columns: constant, Judge 1, Tech, and pre filing value.
ols_design = np.column_stack(
    (
        np.ones(len(analysis)),
        analysis["Treatment_Judge"].to_numpy(dtype=float),
        analysis["Char5_FirmIndustry"].eq("Tech").to_numpy(dtype=float),
        analysis["Char1_FirmValueBeforeFiling"].to_numpy(dtype=float),
    )
)
ols_coefficients = np.linalg.lstsq(
    ols_design, analysis["PercentChange"].to_numpy(dtype=float), rcond=None
)[0]
q6_judge_coefficient = float(ols_coefficients[1])
att_minus_q6 = att - q6_judge_coefficient


### Match Quality ###

# Count each control's matches to distinguish reuse from unique firms.
reuse_counts = matches["ControlFirmNumber"].value_counts()
eligible_controls = int(analysis["Treatment_Judge"].eq(0).sum())
unique_controls_used = int(reuse_counts.size)
controls_used_more_than_once = int(reuse_counts.gt(1).sum())
maximum_control_reuse = int(reuse_counts.max())
share_matches_using_repeated_controls = float(
    matches["ControlFirmNumber"].map(reuse_counts).gt(1).mean()
)

overall_quality = {
    "treated_firms": int(len(matches)),
    "eligible_control_firms": eligible_controls,
    "unique_controls_used": unique_controls_used,
    "unique_controls_used_share": unique_controls_used / eligible_controls,
    "controls_used_more_than_once": controls_used_more_than_once,
    "maximum_control_reuse": maximum_control_reuse,
    "share_matches_using_repeated_controls": share_matches_using_repeated_controls,
    "mean_distance": float(matches["AbsoluteDistance"].mean()),
    "median_distance": float(matches["AbsoluteDistance"].median()),
    "p95_distance": float(matches["AbsoluteDistance"].quantile(0.95)),
    "maximum_distance": float(matches["AbsoluteDistance"].max()),
    "exact_value_matches": int(matches["AbsoluteDistance"].eq(0).sum()),
    "outside_support": int(matches["OutsideSupport"].sum()),
}

# Convert stored distances from millions to thousands of dollars for display.
distance_display_scale = 1000.0


### Write Tables ###

estimate_table = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Nearest Neighbor Matching Estimates}",
    r"\label{tab:q7_matching_estimates}",
    r"\small",
    r"\renewcommand{\arraystretch}{1.12}",
    r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}",
    r"\toprule",
    r"\multicolumn{4}{l}{\textit{Panel A. Matched outcome comparison}} \\",
    r"\addlinespace[3pt]",
    r"Statistic & Matched Judge 0 & Judge 1 & Difference \\",
    r"\midrule",
    f"Mean percent change & {matched_control_mean:.3f} & {treated_mean:.3f} & {att:.3f} " + r"\\",
    f"Matched observations & {len(matches):,} & {len(matches):,} & " + r"\\",
    r"\addlinespace[6pt]",
    r"\multicolumn{4}{l}{\textit{Panel B. Comparison with Question 6}} \\",
    r"\addlinespace[3pt]",
    r"Statistic & Matching & Adjusted OLS & Difference \\",
    r"\midrule",
    f"Judge 1 estimate & {att:.3f} & {q6_judge_coefficient:.3f} & {att_minus_q6:.3f} " + r"\\",
    r"\bottomrule",
    r"\end{tabular*}",
    r"\par\vspace{4pt}",
    r"\begin{minipage}{\linewidth}",
    r"\footnotesize Notes: Each Judge 1 firm is matched with replacement to the Judge 0 firm in the same industry that minimizes the absolute difference in pre filing firm value. Equal distances select the lower control value; duplicate control values select the smallest FirmNumber. The matched Judge 0 mean weights each control by its number of matches, and matched observations count entries rather than unique controls. Panel A reports outcome means in percent and the Judge 1 minus matched Judge 0 difference in percentage points. Panel B reports estimates and the matching minus adjusted OLS difference in percentage points. Differences use unrounded estimates. One observation with zero pre filing value is excluded. Standard errors are not reported.",
    r"\end{minipage}",
    r"\end{table}",
]
(OUTPUT_DIR / "Table_7_Matching_Estimates.tex").write_text(
    "\n".join(estimate_table) + "\n", encoding="utf-8"
)

quality_table = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Overall Matching Quality}",
    r"\label{tab:q7_matching_quality}",
    r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}cccc@{}}",
    r"\toprule",
    r"\multicolumn{4}{l}{\textit{Panel A. Absolute value distance in thousands of dollars}} \\",
    r"\addlinespace[3pt]",
    r"Mean & Median & P95 & Maximum \\",
    r"\midrule",
    f"{distance_display_scale * overall_quality['mean_distance']:,.3f} & "
    f"{distance_display_scale * overall_quality['median_distance']:,.3f} & "
    f"{distance_display_scale * overall_quality['p95_distance']:,.3f} & "
    f"{distance_display_scale * overall_quality['maximum_distance']:,.3f} " + r"\\",
    r"\addlinespace[6pt]",
    r"\multicolumn{4}{l}{\textit{Panel B. Control availability and exact matches}} \\",
    r"\addlinespace[3pt]",
    r"Eligible controls & Unique controls used & Share used & Exact matches \\",
    r"\midrule",
    f"{eligible_controls:,} & {unique_controls_used:,} & "
    f"{100 * overall_quality['unique_controls_used_share']:.3f}\\% & "
    f"{overall_quality['exact_value_matches']:,} " + r"\\",
    r"\addlinespace[6pt]",
    r"\multicolumn{4}{l}{\textit{Panel C. Control reuse and empirical support}} \\",
    r"\addlinespace[3pt]",
    r"Reused controls & Maximum reuse & \shortstack{Matches using\\reused controls} & Outside support \\",
    r"\midrule",
    f"{controls_used_more_than_once:,} & {maximum_control_reuse:,} & "
    f"{100 * share_matches_using_repeated_controls:.3f}\\% & "
    f"{overall_quality['outside_support']:,} " + r"\\",
    r"\bottomrule",
    r"\end{tabular*}",
    r"\par\vspace{4pt}",
    r"\begin{minipage}{\linewidth}",
    r"\footnotesize Notes: Distances are absolute differences in pre filing firm value; P95 denotes the 95th percentile. Exact matches have zero value distance. Share used is the percentage of eligible Judge 0 firms selected at least once. Reused controls are distinct Judge 0 firms selected more than once. Matches using reused controls is the percentage of treated firms whose selected control is used at least twice. Outside support counts treated firms with values below the minimum or above the maximum Judge 0 value in their industry. These firms are retained and matched to the nearest boundary control. Matching uses replacement without a caliper.",
    r"\end{minipage}",
    r"\end{table}",
]
(OUTPUT_DIR / "Table_7_Matching_Quality.tex").write_text(
    "\n".join(quality_table) + "\n", encoding="utf-8"
)

industry_quality_table = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Matching Quality by Industry}",
    r"\label{tab:q7_matching_quality_industry}",
    r"\small",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
    r"\toprule",
    r"Industry & \shortstack{Treated\\firms} & \shortstack{Unique\\controls used} & \multicolumn{3}{c}{Absolute value distance} & \shortstack{Outside\\support} \\",
    r"\cmidrule(lr){4-6}",
    r" & & & Mean & P95 & Maximum & \\",
    r"\midrule",
]

for row in quality_rows:
    industry_quality_table.append(
        f"{row['industry']} & {row['treated_firms']:,} & "
        f"{row['unique_controls_used']:,} & "
        f"{distance_display_scale * row['mean_distance']:,.3f} & "
        f"{distance_display_scale * row['p95_distance']:,.3f} & "
        f"{distance_display_scale * row['maximum_distance']:,.3f} & "
        f"{row['outside_support']:,} \\\\"
    )

industry_quality_table.extend(
    [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\par\vspace{4pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize Notes: Matching is exact on the four industry categories and selects the nearest pre filing value with replacement. Distances are in thousands of dollars; P95 denotes the 95th percentile. Unique controls used counts distinct matched Judge 0 firms within each industry. Outside support counts treated firms whose pre filing value is outside their industry's observed Judge 0 value range. All such firms remain in the matching estimate.",
        r"\end{minipage}",
        r"\end{table}",
    ]
)
(OUTPUT_DIR / "Table_7_Matching_Quality_By_Industry.tex").write_text(
    "\n".join(industry_quality_table) + "\n", encoding="utf-8"
)


### Save Results ###

results = {
    "undefined_percent_change_rows": undefined_change_count,
    "matching_with_replacement": True,
    "distance_unit": "millions of dollars",
    "tie_rule": "Lower control value, then smallest FirmNumber for exact duplicates",
    "treated_mean_percent": treated_mean,
    "matched_control_mean_percent": matched_control_mean,
    "att_percentage_points": att,
    "q6_adjusted_ols_percentage_points": q6_judge_coefficient,
    "att_minus_q6_percentage_points": att_minus_q6,
    "overall_match_quality": overall_quality,
    "match_quality_by_industry": quality_rows,
    "causal_assumptions": [
        "Conditional independence given industry and prefiling value",
        "Adequate common support",
        "Consistent treatment",
        "No interference across firms",
        "Accurate measurement",
        "No unobserved confounding",
    ],
}
(RESULTS_DIR / "Question_7_Results.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)

print(f"Question 7 complete. ATT: {att:.3f} percentage points")
print(f"Question 6 comparison: {q6_judge_coefficient:.3f} percentage points")
print(f"Treated firms outside support: {overall_quality['outside_support']:,}")
