# Judge Assignment

Python analysis of judge assignment and firm value using a simulated corporate finance dataset. The project includes descriptive statistics, distribution plots, group comparisons, OLS regressions, and nearest neighbor matching. Written answers are maintained separately in LaTeX.

## Repository structure

| Directory | Contents |
| --- | --- |
| `code/` | Python scripts, dependencies, and LaTeX sources |
| `data/` | Source CSV dataset |
| `output/` | Generated PNG figures, TeX tables, and the separately compiled report |
| `README/` | Assignment PDF and reproduction guides |
| `tmp/` | Intermediate numerical results, caches, and document build files |

## Data

Place the course dataset at:

```text
data/PhdFinance_FNCE7020_HW1_Data.csv
```

The dataset contains 1,000,000 firms and eight variables, with one observation per firm and no missing source values. It is a cross sectional simulated dataset supplied with the assignment. The scripts never modify the source CSV.

| Variable | Description |
| --- | --- |
| `FirmNumber` | Unique firm identifier |
| `Char1_FirmValueBeforeFiling` | Firm value before filing, in millions of dollars |
| `Char2_FirmProfitMarginBeforeFiling` | Profit margin before filing, as a decimal ratio |
| `Char3_FirmInterestCoverageBeforeFiling` | Interest coverage before filing |
| `Char4_FirmLeverageBeforeFiling` | Debt share before filing, as a decimal ratio |
| `Char5_FirmIndustry` | Industry before filing |
| `Treatment_Judge` | Assigned judge, coded zero or one |
| `Outcome_FirmValue1YearAfterFiling` | Firm value one year after filing, in millions of dollars |

Source CSV SHA256:

```text
dc87cc6ccb0119ac20478ed7b28acfcf160abeab44529a73c8718941933cf4e1
```

The assignment is stored at `README/HW1_FNCE7020_Fall2026.pdf`. It provides the questions and data description but is not loaded by the Python scripts.

## Reproduce the analysis

Use Python 3.9 or newer. Run the following commands from the project root, which contains the `code/` and `data/` directories:

```sh
python3 -m pip install -r code/requirements.txt
python3 code/00_Main.py
```

The main script runs Questions 1 through 7 in order and generates only analysis results. Each question has a separate script and can also run independently, for example:

```sh
python3 code/06_Question_6.py
```

Question 8 is a conceptual discussion maintained directly in `code/HW1_Concise_Report.tex`; it has no Python script or generated table. The main script does not create or compile the report or any README format.

The dependency file specifies minimum package versions rather than an exactly pinned environment.

## Build the report separately

Written answers are maintained in `code/HW1_Concise_Report.tex`. This source references the generated TeX tables and PNG figures. If the analysis changes, update numerical statements in the report manually before compiling.

A TeX installation with `latexmk` and `pdflatex` is required. After running the analysis, execute these commands from the project root:

```sh
mkdir -p tmp/documents
latexmk -pdf -outdir=tmp/documents code/HW1_Concise_Report.tex
cp tmp/documents/HW1_Concise_Report.pdf output/HW1_Concise_Report.pdf
```

The report includes Questions 1 through 8, with each question starting on a new page. The first page reserves a place for the GitHub repository link.

## Outputs

The `output/` directory contains:

- Nine PNG figures.
- Eleven generated TeX tables.
- The separately compiled `HW1_Concise_Report.pdf`.

Machine readable CSV and JSON results are stored in `tmp/results/`. Displayed statistics use three decimal places, counts use integers, and very small p values are displayed as `<0.001`. Saved numerical results retain full precision.

## Analytical choices

- Percentage change is calculated as `100 * (after - before) / before`.
- Firm 686814 has zero value before and after filing. Its percentage change is undefined, so analyses using this outcome exclude that observation. Raw summaries retain it.
- Question 3 compares continuous means using two sided Welch tests. Its density plots use common bin edges and normalize each judge group to unit area.
- Questions 4 and 6 report conventional homoskedastic OLS standard errors. Question 6 controls for Tech status and pre filing firm value.
- Question 7 matches each Judge 1 firm to the closest Judge 0 firm in the same industry using pre filing value. Matching uses replacement. Equal distances select the lower control value; duplicate control values select the smallest firm identifier.
- Treated firms outside their industry's observed control value range remain in the matching sample and are matched to the nearest boundary control.
- No winsorization, trimming, imputation, or random sampling affects the reported estimates.

## Additional documentation

The reproduction guide is also available as `README/README.html` and `README/README.pdf`. The PDF guide is maintained in `code/README.tex` and can be rebuilt separately:

```sh
latexmk -pdf -outdir=tmp/documents code/README.tex
cp tmp/documents/README.pdf README/README.pdf
```

Review the code and results before submission. The assignment requires the submitted work to be understood and defensible by its author.
