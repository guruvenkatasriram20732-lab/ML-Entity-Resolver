"""
Feature engineering module for Amazon ML Challenge - Business Entity Resolution.
Computes string distances, token overlaps, geographic consistency, and TF-IDF cosine similarities.
"""

from collections import Counter
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

def fast_levenshtein(s1: str, s2: str, max_len: int = 100) -> int:
    s1, s2 = s1[:max_len], s2[:max_len]
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]

def fast_jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_dist = max(len1, len2) // 2 - 1
    s1_m, s2_m = [False] * len1, [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if not s2_m[j] and s1[i] == s2[j]:
                s1_m[i] = s2_m[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    t, k = 0, 0
    for i in range(len1):
        if not s1_m[i]:
            continue
        while not s2_m[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    jaro = (matches / len1 + matches / len2 + (matches - t // 2) / matches) / 3.0
    p = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            p += 1
            if p == 4:
                break
        else:
            break
    return jaro + p * 0.1 * (1.0 - jaro)

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text:
        return set()
    padded = f"  {text}  "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}

class FastTfidfEngine:
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.idf: Dict[str, float] = {}

    def fit(self, texts: List[str]) -> None:
        df_counts: Counter = Counter()
        N = len(texts)
        for t in texts:
            df_counts.update(get_char_ngrams(t, 3))
        for term, cnt in df_counts.most_common(self.max_features):
            self.idf[term] = math.log((1 + N) / (1 + cnt)) + 1.0

    def get_sparse_vector(self, text: str) -> Dict[str, float]:
        ngrams = get_char_ngrams(text, 3)
        if not ngrams:
            return {}
        counts = Counter(ngrams)
        vec, norm_sq = {}, 0.0
        for term, cnt in counts.items():
            if term in self.idf:
                v = cnt * self.idf[term]
                vec[term] = v
                norm_sq += v * v
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for k in vec:
                vec[k] /= norm
        return vec

    @staticmethod
    def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        return sum(val * v2.get(k, 0.0) for k, val in v1.items())

class FeatureExtractor:
    FEATURE_NAMES = [
        "name_levenshtein_sim", "name_jaro_winkler", "name_token_jaccard", "name_token_containment",
        "name_char_3gram_jaccard", "name_exact_match", "name_prefix_match", "name_first_token_match",
        "addr_levenshtein_sim", "addr_token_jaccard", "addr_number_match", "city_match", "state_match",
        "zip_match", "country_match", "tfidf_cosine_sim", "len_diff", "len_ratio", "token_diff",
        "is_source2", "is_source3"
    ]

    def __init__(self, tfidf_engine: Optional[FastTfidfEngine] = None):
        self.tfidf = tfidf_engine or FastTfidfEngine()
        self.tfidf_cache: Dict[str, Dict[str, float]] = {}

    def precompute_tfidf_cache(self, records: List[dict]) -> None:
        for r in records:
            if r["id"] not in self.tfidf_cache:
                self.tfidf_cache[r["id"]] = self.tfidf.get_sparse_vector(r.get("full_text", ""))

    def extract_pair_features(self, r1: dict, r2: dict) -> List[float]:
        n1, n2 = r1.get("norm_name", ""), r2.get("norm_name", "")
        max_n = max(len(n1), len(n2))
        name_lev = 1.0 - (fast_levenshtein(n1, n2) / max_n) if max_n > 0 else 1.0
        name_jw = fast_jaro_winkler(n1, n2)

        t1, t2 = set(r1.get("name_tokens", [])), set(r2.get("name_tokens", []))
        if t1 and t2:
            token_jacc = len(t1 & t2) / len(t1 | t2)
            token_cont = len(t1 & t2) / min(len(t1), len(t2))
        else:
            token_jacc = token_cont = 1.0 if not t1 and not t2 else 0.0

        cg1, cg2 = get_char_ngrams(n1, 3), get_char_ngrams(n2, 3)
        char_jacc = len(cg1 & cg2) / len(cg1 | cg2) if cg1 and cg2 else (1.0 if not cg1 and not cg2 else 0.0)

        name_exact = 1.0 if n1 and n1 == n2 else 0.0
        p1, p2 = n1[:3], n2[:3]
        prefix_match = 1.0 if p1 and p1 == p2 else 0.0

        toks1, toks2 = r1.get("name_tokens", []), r2.get("name_tokens", [])
        first_match = 1.0 if toks1 and toks2 and toks1[0] == toks2[0] else 0.0

        a1, a2 = r1.get("norm_address", ""), r2.get("norm_address", "")
        max_a = max(len(a1), len(a2))
        addr_lev = 1.0 - (fast_levenshtein(a1, a2) / max_a) if max_a > 0 else 0.5

        at1, at2 = set(r1.get("addr_tokens", [])), set(r2.get("addr_tokens", []))
        addr_jacc = len(at1 & at2) / len(at1 | at2) if at1 and at2 else 0.5

        num1, num2 = set(r1.get("addr_numbers", [])), set(r2.get("addr_numbers", []))
        addr_num_match = 1.0 if (num1 and num2 and (num1 & num2)) else (-1.0 if (num1 and num2) else 0.0)

        c1, c2 = r1.get("norm_city", ""), r2.get("norm_city", "")
        city_match = 1.0 if c1 and c1 == c2 else (-1.0 if c1 and c2 else 0.0)

        s1_st, s2_st = r1.get("norm_state", ""), r2.get("norm_state", "")
        state_match = 1.0 if s1_st and s1_st == s2_st else (-1.0 if s1_st and s2_st else 0.0)

        z1, z2 = r1.get("norm_zip", ""), r2.get("norm_zip", "")
        zip_match = 1.0 if z1 and z1 == z2 else (-1.0 if z1 and z2 else 0.0)

        cnt1, cnt2 = r1.get("norm_country", ""), r2.get("norm_country", "")
        country_match = 1.0 if cnt1 and cnt1 == cnt2 else (-1.0 if cnt1 and cnt2 else 0.0)

        v1 = self.tfidf_cache.get(r1["id"], self.tfidf.get_sparse_vector(r1.get("full_text", "")))
        v2 = self.tfidf_cache.get(r2["id"], self.tfidf.get_sparse_vector(r2.get("full_text", "")))
        tfidf_sim = FastTfidfEngine.cosine_similarity(v1, v2)

        return [
            name_lev, name_jw, token_jacc, token_cont, char_jacc, name_exact, prefix_match,
            first_match, addr_lev, addr_jacc, addr_num_match, city_match, state_match,
            zip_match, country_match, tfidf_sim, abs(len(n1) - len(n2)),
            (min(len(n1), len(n2)) / max_n) if max_n > 0 else 1.0,
            abs(len(toks1) - len(toks2)),
            1.0 if r2.get("source_tag") == "s2" else 0.0,
            1.0 if r2.get("source_tag") == "s3" else 0.0,
        ]

    def build_feature_matrix(self, pairs: List[Tuple[str, str]], l1: Dict[str, dict], l2: Dict[str, dict]) -> np.ndarray:
        return np.array([self.extract_pair_features(l1[s], l2[t]) for s, t in pairs], dtype=np.float32)
