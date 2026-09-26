@echo off
echo === Installing dependencies ===
pip install -r requirements.txt

echo === Running Business Entity Resolution Pipeline ===
python main.py

echo === Pipeline Completed! ===
echo Check output\matching_results.tsv and output\candidate_pairs.tsv
pause
