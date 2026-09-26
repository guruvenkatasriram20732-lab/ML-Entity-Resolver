"""
Preprocessing and normalization module for Amazon ML Challenge - Business Entity Resolution.
"""

import re
import unicodedata
from typing import Dict, List, Set
import pandas as pd

COUNTRY_ALIASES: Dict[str, str] = {
    "us": "US", "usa": "US", "u.s.": "US", "united states": "US", "america": "US",
    "in": "IN", "ind": "IN", "india": "IN", "bharat": "IN",
    "fr": "FR", "fra": "FR", "france": "FR",
    "uk": "GB", "gb": "GB", "united kingdom": "GB",
    "ca": "CA", "can": "CA", "canada": "CA",
    "de": "DE", "deu": "DE", "germany": "DE",
    "au": "AU", "aus": "AU", "australia": "AU",
    "sg": "SG", "singapore": "SG",
    "jp": "JP", "japan": "JP",
}

ENTITY_SUFFIXES: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "ltd", "limited",
    "co", "company", "pvt", "private", "gmbh", "sarl", "sa", "bhd", "plc"
}

ADDRESS_EXPANSIONS: Dict[str, str] = {
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "pkwy": "parkway",
    "ct": "court",
    "pl": "place",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"
}


class Preprocessor:
    @staticmethod
    def strip_accents(text: str) -> str:
        if not text:
            return ""
        norm = unicodedata.normalize("NFKD", str(text))
        return "".join(c for c in norm if not unicodedata.combining(c))

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text or pd.isna(text):
            return ""
        text = Preprocessor.strip_accents(str(text).lower()).replace("&", " and ")
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    @classmethod
    def normalize_name(cls, name: str) -> str:
        tokens = cls.normalize_text(name).split()
        if not tokens:
            return ""
        while tokens and tokens[-1] in ENTITY_SUFFIXES:
            tokens.pop()
        return " ".join(tokens) if tokens else cls.normalize_text(name)

    @classmethod
    def normalize_address(cls, address: str) -> str:
        tokens = cls.normalize_text(address).split()
        return " ".join([ADDRESS_EXPANSIONS.get(t, t) for t in tokens])

    @staticmethod
    def normalize_country(country: str) -> str:
        if not country or pd.isna(country):
            return ""
        clean = Preprocessor.strip_accents(str(country).lower().strip())
        clean = re.sub(r"[^\w\s]", "", clean).strip()
        if clean in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[clean]
        if len(clean) in (2, 3) and clean.isalpha():
            return clean.upper()
        return clean.upper()

    @staticmethod
    def normalize_zip(zip_code: str) -> str:
        if not zip_code or pd.isna(zip_code):
            return ""
        return re.sub(r"[^\w]", "", str(zip_code).strip().upper())

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        return re.findall(r"\b\d+\b", text) if text else []

    @classmethod
    def preprocess_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["norm_name"] = df["name"].apply(cls.normalize_name)
        df["norm_address"] = df["address"].apply(cls.normalize_address)
        df["norm_city"] = df["city"].apply(cls.normalize_text)
        df["norm_state"] = df["state"].apply(cls.normalize_text)
        df["norm_country"] = df["country"].apply(cls.normalize_country)
        df["norm_zip"] = df["zip"].apply(cls.normalize_zip)
        df["name_tokens"] = df["norm_name"].apply(lambda s: s.split())
        df["addr_tokens"] = df["norm_address"].apply(lambda s: s.split())
        df["addr_numbers"] = df["norm_address"].apply(cls.extract_numbers)
        df["full_text"] = (
            df["norm_name"] + " " +
            df["norm_address"] + " " +
            df["norm_city"] + " " +
            df["norm_country"]
        ).str.strip()
        return df
