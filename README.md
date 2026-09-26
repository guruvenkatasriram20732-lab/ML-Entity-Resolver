# Amazon ML Challenge – Business Entity Resolution

An enterprise-grade, memory-efficient Python pipeline designed to resolve and link business entities across **Source 1**, **Source 2**, and **Source 3** without any external APIs, web scrapers, or third-party business directories.

---

## ⚡ Quick Start (VS Code or Terminal)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note: The codebase contains a dual-engine architecture: it utilizes `scikit-learn` when available, and automatically falls back to an internal zero-dependency pure-NumPy regularized Logistic Regression classifier and sparse TF-IDF engine if scikit-learn is not installed).*

### 2. Place Your Dataset Files
Place your 7 TSV dataset files inside the `dataset/` folder:
```text
Amazon_ML_Challenge/
└── dataset/
    ├── train_source1.tsv
    ├── train_source2.tsv
    ├── train_source3.tsv
    ├── train_ground_truth.tsv
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```
*(If you run the pipeline without placing files, it automatically bootstraps a realistic benchmark dataset so you can test end-to-end immediately).*

### 3. Run the Pipeline
```bash
python main.py
```

### 4. Output Files Generated
Upon completion, the pipeline outputs:
- `output/matching_results.tsv`: Exact official submission format. Every test Source 1 entity appears exactly once, followed by tab and comma-separated matched Source 2/3 IDs (or empty value if no match).
- `output/candidate_pairs.tsv`: All candidate pairs with their match probabilities.
- `output/validation_metrics.json`: Precision, Recall, and tuned **F0.5** score table.

---

## 🏗️ Architecture & Memory Optimization

To handle **~1GB datasets** without out-of-memory errors:
1. **Multi-Key Inverted Blocking**: Instead of generating an $O(N_1 \times (N_2 + N_3))$ Cartesian product (billions of pairs), inverted indices are indexed by `(Country, 3-char prefix)`, `(Country, First Token)`, `(Country, Postal Code)`, and `(Country, Longest Token)`.
2. **Frequency Pruning & Candidate Capping**: Overly common tokens (e.g. `the`, `ltd`, `inc`) are pruned, and candidates are capped at $K \le 60$ per query record.
3. **Sparse 3-gram TF-IDF**: Uses integer counts and sparse vector dot products for sub-millisecond cosine similarity calculations.
4. **Group Validation Split**: 80/20 train/val split split strictly on Source 1 business entities to eliminate data leakage.
5. **F0.5 Metric Optimization**: Sweeps classification threshold over $[0.20, 0.95]$ to maximize $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$, emphasizing Precision twice as heavily as Recall.
