"""Run all homework analyses and generate their results."""

from pathlib import Path
import subprocess
import sys


### Paths ###

# Resolve paths from this script, not the working directory.
ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = ROOT / "tmp" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


### Run analyses ###

# Question 8 is written directly in LaTeX and needs no script.
scripts = [CODE_DIR / f"{number:02d}_Question_{number}.py" for number in range(1, 8)]

for script in scripts:
    print(f"Running {script.name}", flush=True)
    # Use the current Python interpreter and stop on failure.
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

print("All analyses completed successfully.")
