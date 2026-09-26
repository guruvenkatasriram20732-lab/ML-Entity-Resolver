"""
Inference and output generator module for Amazon ML Challenge - Business Entity Resolution.
Generates candidate_pairs.tsv and matching_results.tsv conforming strictly to competition specs.
"""

from collections import defaultdict
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

class Predictor:
    def __init__(self, model, feature_extractor, blocking_engine, threshold: float):
        self.model = model
        self.fe = feature_extractor
        self.blocking = blocking_engine
        self.threshold = threshold

    def predict_test_dataset(self, test_s1_df: pd.DataFrame, test_target_df: pd.DataFrame, gt_header: Tuple[str, str], candidate_file: Path, results_file: Path):
        self.blocking.index_target_entities(test_target_df)
        t_records = test_target_df.to_dict(orient="records")
        t_lookup = {r["id"]: r for r in t_records}
        s1_records = test_s1_df.to_dict(orient="records")
        s1_lookup = {r["id"]: r for r in s1_records}
        all_s1_ids = [r["id"] for r in s1_records]

        self.fe.precompute_tfidf_cache(s1_records)
        self.fe.precompute_tfidf_cache(t_records)
        pairs = self.blocking.generate_candidate_pairs(test_s1_df)
        probs = self.model.predict_proba(self.fe.build_feature_matrix(pairs, s1_lookup, t_lookup))[:, 1] if pairs else np.array([])

        candidate_file.parent.mkdir(parents=True, exist_ok=True)
        with open(candidate_file, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source1_id", "candidate_id", "score"])
            for (s1_id, c_id), score in zip(pairs, probs):
                w.writerow([s1_id, c_id, f"{score:.5f}"])

        matched: Dict[str, List[str]] = defaultdict(list)
        for (s1_id, c_id), score in zip(pairs, probs):
            if score >= self.threshold and c_id in t_lookup:
                matched[s1_id].append(c_id)

        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([gt_header[0], gt_header[1]])
            for s1_id in all_s1_ids:
                matches = sorted(list(set(matched.get(s1_id, []))))
                w.writerow([s1_id, ",".join(matches) if matches else ""])
