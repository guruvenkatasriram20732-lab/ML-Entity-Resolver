"""
Multi-key inverted index blocking module for Amazon ML Challenge.
Avoids O(N^2) Cartesian join using inverted indexes, frequent-token pruning,
and fast lexical pre-scoring to support ~1GB datasets within memory constraints.
"""

from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from src.config import INVERTED_INDEX_MAX_POSTINGS, MAX_CANDIDATES_PER_SOURCE1

logger = logging.getLogger(__name__)


class BlockGenerator:
    STOPWORDS = {
        "the", "and", "of", "in", "at", "for", "on", "by", "a", "an", "to",
        "llc", "inc", "ltd", "corp", "co", "company", "pvt", "limited"
    }

    @classmethod
    def get_blocking_keys(cls, row: dict) -> List[str]:
        keys = []
        c = row.get("norm_country", "").strip()
        name = row.get("norm_name", "").strip()
        zip_c = row.get("norm_zip", "").strip()
        tokens = [t for t in row.get("name_tokens", []) if t not in cls.STOPWORDS and len(t) >= 2]

        if len(name) >= 3:
            pre = name[:3]
            if c:
                keys.append(f"C_PRE::{c}::{pre}")
            keys.append(f"PRE::{pre}")

        if tokens:
            t0 = tokens[0]
            if len(t0) >= 3:
                if c:
                    keys.append(f"C_TOK::{c}::{t0}")
                keys.append(f"TOK::{t0}")

        if zip_c and len(zip_c) >= 3:
            if c:
                keys.append(f"C_ZIP::{c}::{zip_c}")
            keys.append(f"ZIP::{zip_c}")

        if len(tokens) >= 2:
            longest = max(tokens, key=len)
            if len(longest) >= 4:
                if c:
                    keys.append(f"C_LONG::{c}::{longest}")
                keys.append(f"LONG::{longest}")

        return keys


class BlockingEngine:
    def __init__(self, max_candidates: int = MAX_CANDIDATES_PER_SOURCE1):
        self.max_candidates = max_candidates
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        self.target_lookup: Dict[str, dict] = {}

    def index_target_entities(self, target_df: pd.DataFrame) -> None:
        self.inverted_index.clear()
        self.target_lookup.clear()
        records = target_df.to_dict(orient="records")
        for rec in records:
            t_id = rec["id"]
            self.target_lookup[t_id] = rec
            for k in BlockGenerator.get_blocking_keys(rec):
                self.inverted_index[k].append(t_id)

        for k in list(self.inverted_index.keys()):
            if len(self.inverted_index[k]) > INVERTED_INDEX_MAX_POSTINGS:
                del self.inverted_index[k]
        logger.info(f"Target index ready with {len(self.inverted_index)} active blocks")

    def retrieve_candidates_for_record(self, q_rec: dict) -> List[str]:
        keys = BlockGenerator.get_blocking_keys(q_rec)
        counts: Dict[str, int] = defaultdict(int)
        for k in keys:
            for t_id in self.inverted_index.get(k, []):
                counts[t_id] += 1
        if not counts:
            return []
        if len(counts) <= self.max_candidates:
            return list(counts.keys())

        q_tokens = set(q_rec.get("name_tokens", []))
        scored = []
        for t_id, hit_count in counts.items():
            t_rec = self.target_lookup.get(t_id)
            if not t_rec:
                continue
            t_toks = set(t_rec.get("name_tokens", []))
            jacc = len(q_tokens & t_toks) / max(len(q_tokens | t_toks), 1)
            scored.append((hit_count * 2.0 + jacc * 5.0, t_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t_id for _, t_id in scored[:self.max_candidates]]

    def generate_candidate_pairs(
        self,
        source1_df: pd.DataFrame,
        positive_pairs: Optional[Set[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for q_rec in source1_df.to_dict(orient="records"):
            s1_id = q_rec["id"]
            for c_id in self.retrieve_candidates_for_record(q_rec):
                pairs.add((s1_id, c_id))
        if positive_pairs:
            for s1_id, t_id in positive_pairs:
                if t_id in self.target_lookup:
                    pairs.add((s1_id, t_id))
        return list(pairs)
