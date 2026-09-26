"""
Configuration module for Amazon ML Challenge - Business Entity Resolution.
Contains directory paths, model hyperparameters, blocking limits, and feature settings.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Dataset filenames
TRAIN_SOURCE1 = "train_source1.tsv"
TRAIN_SOURCE2 = "train_source2.tsv"
TRAIN_SOURCE3 = "train_source3.tsv"
TRAIN_GROUND_TRUTH = "train_ground_truth.tsv"

TEST_SOURCE1 = "test_source1.tsv"
TEST_SOURCE2 = "test_source2.tsv"
TEST_SOURCE3 = "test_source3.tsv"

# Output filenames
OUTPUT_MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"
OUTPUT_METRICS = OUTPUT_DIR / "validation_metrics.json"

# Blocking & Memory Limits for ~1GB data
MAX_CANDIDATES_PER_SOURCE1 = 60
BLOCKING_NGRAM_SIZE = 3
TFIDF_MAX_FEATURES = 10000
INVERTED_INDEX_MAX_POSTINGS = 1500
LEVENSHTEIN_MAX_LEN = 100

# Evaluation & Metrics
VAL_SPLIT_RATIO = 0.20
RANDOM_STATE = 42
F_BETA = 0.5  # Challenge metric is F0.5
DEFAULT_THRESHOLD = 0.60
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
