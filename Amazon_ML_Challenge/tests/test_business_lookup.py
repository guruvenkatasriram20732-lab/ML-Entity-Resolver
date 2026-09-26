from pathlib import Path

from src.business_lookup import find_similar_businesses


def test_find_similar_businesses_returns_top_matches():
    base = Path(__file__).resolve().parents[1]
    results = find_similar_businesses("Starbucks", base / "dataset")

    assert results
    assert results[0]["business_name"]
    assert "score" in results[0]
    assert "is_match" in results[0]
    assert isinstance(results[0]["score"], float)
