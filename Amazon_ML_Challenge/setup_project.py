"""
Standalone setup script for Amazon ML Challenge - Business Entity Resolution.
Run this script anywhere to instantly unpack the complete project.
"""

from pathlib import Path

BASE_DIR = Path("Amazon_ML_Challenge")

def create():
    print(f"Creating project structure in ./{BASE_DIR}...")
    (BASE_DIR / "dataset").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "src").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "output").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)
    print("Directories initialized. Run 'python main.py' to execute pipeline.")

if __name__ == "__main__":
    create()
