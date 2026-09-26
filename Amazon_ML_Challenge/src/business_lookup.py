from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import DATA_DIR, DEFAULT_THRESHOLD
from src.feature_engineering import FeatureExtractor, FastTfidfEngine
from src.preprocessor import Preprocessor


def _load_business_records(data_dir: Path) -> List[dict]:
    data_dir = Path(data_dir)
    record_frames: List[pd.DataFrame] = []
    file_map = {
        "source1": ["train_source1.tsv", "test_source1.tsv"],
        "source2": ["train_source2.tsv", "test_source2.tsv"],
        "source3": ["train_source3.tsv", "test_source3.tsv"],
    }

    for source_tag, file_names in file_map.items():
        for file_name in file_names:
            path = data_dir / file_name
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            if "name" not in df.columns:
                continue
            df = df.fillna("")
            df["source_tag"] = source_tag
            record_frames.append(df)

    if not record_frames:
        return []

    combined = pd.concat(record_frames, ignore_index=True)
    combined = combined.fillna("")
    preprocessed = Preprocessor.preprocess_dataframe(combined)
    return preprocessed.to_dict(orient="records")


def _make_query_record(name: str) -> dict:
    df = pd.DataFrame([
        {
            "id": "query",
            "name": name,
            "address": "",
            "city": "",
            "state": "",
            "zip": "",
            "country": "",
        }
    ])
    return Preprocessor.preprocess_dataframe(df).iloc[0].to_dict()


def _score_business_match(query_record: dict, candidate_record: dict) -> float:
    feature_extractor = FeatureExtractor(FastTfidfEngine())
    feature_extractor.precompute_tfidf_cache([candidate_record])

    vector = feature_extractor.extract_pair_features(query_record, candidate_record)
    name_jw = max(0.0, min(1.0, vector[1]))
    token_jacc = max(0.0, min(1.0, vector[2]))
    char_jacc = max(0.0, min(1.0, vector[4]))
    addr_jacc = max(0.0, min(1.0, vector[9]))
    city_match = max(0.0, min(1.0, vector[11]))
    state_match = max(0.0, min(1.0, vector[12]))
    country_match = max(0.0, min(1.0, vector[14]))
    tfidf_sim = max(0.0, min(1.0, vector[15]))

    name_similarity = 0.40 * name_jw + 0.25 * token_jacc + 0.20 * char_jacc + 0.15 * tfidf_sim
    address_similarity = 0.45 * addr_jacc + 0.20 * city_match + 0.20 * state_match + 0.15 * country_match
    score = 0.70 * name_similarity + 0.30 * address_similarity
    return float(max(0.0, min(1.0, score)))


def find_similar_businesses(
    business_name: str,
    dataset_dir: str | Path = DATA_DIR,
    limit: int = 5,
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Return top candidate businesses that look similar to the given business name."""
    if not business_name or not business_name.strip():
        return []

    search_dir = Path(dataset_dir)
    records = _load_business_records(search_dir)
    if not records:
        return []

    query_record = _make_query_record(business_name)
    threshold = float(DEFAULT_THRESHOLD if threshold is None else threshold)

    ranked = []
    for record in records:
        if record.get("id", "") == "query":
            continue
        score = _score_business_match(query_record, record)
        ranked.append({
            "business_name": str(record.get("name", "")).strip(),
            "source": str(record.get("source_tag", "")).strip(),
            "score": round(score, 4),
            "is_match": score >= threshold,
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]
