
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
scripts = [
    ROOT / "src" / "eda_quality.py",
    ROOT / "src" / "statistical_analysis.py",
    ROOT / "src" / "train_evaluate.py",
    ROOT / "src" / "visualize.py",
]
for script in scripts:
    print(f"\n>>> Running {script.name}")
    subprocess.run([sys.executable, str(script)], check=True)

print("\nPipeline completed successfully.")
