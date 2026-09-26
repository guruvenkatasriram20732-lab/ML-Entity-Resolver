"""
Data loader module for Amazon ML Challenge - Business Entity Resolution.
Handles robust loading, column auto-detection, schema normalization, and ground-truth parsing.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    ID_CANDIDATES = ["id", "source1_id", "source2_id", "source3_id", "record_id", "business_id", "s1_id", "s2_id", "s3_id"]
    NAME_CANDIDATES = ["name", "business_name", "company_name", "title", "legal_name", "store_name"]
    ADDRESS_CANDIDATES = ["address", "street_address", "street", "addr", "location", "address_line_1"]
    CITY_CANDIDATES = ["city", "town", "locality", "municipality"]
    STATE_CANDIDATES = ["state", "province", "region", "state_code"]
    ZIP_CANDIDATES = ["zip", "postal_code", "zipcode", "postcode", "pincode"]
    COUNTRY_CANDIDATES = ["country", "country_code", "cntry", "nation", "iso_country"]
    PHONE_CANDIDATES = ["phone", "phone_number", "telephone", "tel", "contact"]
    CATEGORY_CANDIDATES = ["category", "categories", "business_category", "type", "industry"]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.ground_truth_columns: Tuple[str, str] = ("source1_id", "matched_id")

    @staticmethod
    def _read_tsv(filepath: Path) -> pd.DataFrame:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            df = pd.read_csv(
                filepath,
                sep="\t",
                dtype=str,
                encoding="utf-8",
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL,
                na_filter=False
            )
        except Exception:
            df = pd.read_csv(
                filepath,
                sep="\t",
                dtype=str,
                encoding="latin1",
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
                na_filter=False
            )
        return df.apply(lambda col: col.str.strip() if hasattr(col, "str") else col)

    def _detect_column(self, df: pd.DataFrame, candidate_names: List[str]) -> Optional[str]:
        col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
        for cand in candidate_names:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    def standardize_dataframe(self, df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
        std_df = pd.DataFrame()

        id_col = self._detect_column(df, self.ID_CANDIDATES) or df.columns[0]
        std_df["id"] = df[id_col].astype(str)

        name_col = self._detect_column(df, self.NAME_CANDIDATES)
        std_df["name"] = df[name_col].astype(str) if name_col else ""

        addr_col = self._detect_column(df, self.ADDRESS_CANDIDATES)
        std_df["address"] = df[addr_col].astype(str) if addr_col else ""

        city_col = self._detect_column(df, self.CITY_CANDIDATES)
        std_df["city"] = df[city_col].astype(str) if city_col else ""

        state_col = self._detect_column(df, self.STATE_CANDIDATES)
        std_df["state"] = df[state_col].astype(str) if state_col else ""

        zip_col = self._detect_column(df, self.ZIP_CANDIDATES)
        std_df["zip"] = df[zip_col].astype(str) if zip_col else ""

        country_col = self._detect_column(df, self.COUNTRY_CANDIDATES)
        std_df["country"] = df[country_col].astype(str) if country_col else ""

        phone_col = self._detect_column(df, self.PHONE_CANDIDATES)
        std_df["phone"] = df[phone_col].astype(str) if phone_col else ""

        cat_col = self._detect_column(df, self.CATEGORY_CANDIDATES)
        std_df["category"] = df[cat_col].astype(str) if cat_col else ""

        std_df["source_tag"] = source_tag

        for col in ["name", "address", "city", "state", "zip", "country", "phone", "category"]:
            std_df[col] = std_df[col].replace({"nan": "", "None": "", "NULL": "", "null": ""})

        return std_df

    def load_source_file(self, filename: str, source_tag: str) -> pd.DataFrame:
        df = self._read_tsv(self.data_dir / filename)
        logger.info(f"Loaded {filename}: {len(df)} records")
        return self.standardize_dataframe(df, source_tag)

    def load_ground_truth(self, filename: str) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]]]:
        df = self._read_tsv(self.data_dir / filename)
        cols = list(df.columns)
        if len(cols) >= 2:
            self.ground_truth_columns = (cols[0], cols[1])

        gt_map: Dict[str, Set[str]] = {}
        positive_pairs: Set[Tuple[str, str]] = set()

        if len(cols) == 2:
            s1_col, match_col = cols[0], cols[1]
            for _, row in df.iterrows():
                s1_id = str(row[s1_col]).strip()
                match_val = str(row[match_col]).strip()
                if not s1_id:
                    continue
                if s1_id not in gt_map:
                    gt_map[s1_id] = set()

                if not match_val or match_val.lower() in ("nan", "none", "null"):
                    continue

                tokens = [t.strip() for t in match_val.replace(";", ",").replace("|", ",").split(",") if t.strip()]
                for m_id in tokens:
                    if m_id.lower() not in ("nan", "none", "null"):
                        gt_map[s1_id].add(m_id)
                        positive_pairs.add((s1_id, m_id))

        elif len(cols) >= 3:
            s1_col = cols[0]
            target_cols = cols[1:]
            for _, row in df.iterrows():
                s1_id = str(row[s1_col]).strip()
                if not s1_id:
                    continue
                if s1_id not in gt_map:
                    gt_map[s1_id] = set()
                for c in target_cols:
                    val = str(row[c]).strip()
                    if val and val.lower() not in ("nan", "none", "null"):
                        gt_map[s1_id].add(val)
                        positive_pairs.add((s1_id, val))

        logger.info(f"Loaded ground truth: {len(gt_map)} query entities, {len(positive_pairs)} match pairs")
        return gt_map, positive_pairs
