#!/usr/bin/env python3
"""
Master pipeline runner for Amazon ML Challenge - Business Entity Resolution.
Executes loading, normalization, candidate blocking, feature extraction,
ML model training, F0.5 threshold tuning, test prediction, and official validation.
"""

import argparse
import csv
import logging
import os
from pathlib import Path
import sys
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    DEFAULT_THRESHOLD,
    LOG_FORMAT,
    MODEL_DIR,
    OUTPUT_CANDIDATE_PAIRS,
    OUTPUT_MATCHING_RESULTS,
    OUTPUT_METRICS,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    VAL_SPLIT_RATIO,
)
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.blocking import BlockingEngine
from src.feature_engineering import FastTfidfEngine, FeatureExtractor
from src.model import EntityResolutionModel
from src.evaluate import calculate_metrics, save_metrics_report, split_entities, tune_threshold
from src.predictor import Predictor
from src.validator import SubmissionValidator
from src.business_lookup import find_similar_businesses

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AmazonMLChallenge")


def create_sample_dataset(data_dir: Path) -> None:
    """Generates a realistic multi-country benchmark dataset if custom files are not yet placed."""
    logger.info(f"Generating realistic sample dataset in {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_s1 = [
        ["s1_001", "Starbucks Coffee Inc", "102 Pike St", "Seattle", "WA", "98101", "United States"],
        ["s1_002", "Boulangerie Paul", "28 Rue de Rivoli", "Paris", "Ile-de-France", "75004", "France"],
        ["s1_003", "Infosys Limited", "Electronics City Hosur Rd", "Bengaluru", "Karnataka", "560100", "India"],
        ["s1_004", "McDonald's Restaurant", "110 N Carpenter St", "Chicago", "IL", "60607", "USA"],
        ["s1_005", "Tata Consultancy Services Ltd", "Nirmal Building Nariman Point", "Mumbai", "Maharashtra", "400021", "India"],
        ["s1_006", "Carrefour Market", "73 Boulevard Saint-Germain", "Paris", "Paris", "75005", "France"],
        ["s1_007", "Apple Store", "1 Infinite Loop", "Cupertino", "CA", "95014", "US"],
        ["s1_008", "Wipro Technologies", "Doddakannelli Sarjapur Road", "Bangalore", "Karnataka", "560035", "India"],
        ["s1_009", "Chez Janou Bistro", "2 Rue Roger Verlomme", "Paris", "Paris", "75003", "France"],
        ["s1_010", "Target Corporation", "1000 Nicollet Mall", "Minneapolis", "MN", "55403", "USA"],
        ["s1_011", "Reliance Retail Ltd", "Maker Chambers IV Nariman Point", "Mumbai", "Maharashtra", "400021", "India"],
        ["s1_012", "Subway Sandwiches", "200 5th Ave", "New York", "NY", "10010", "US"],
        ["s1_013", "Le Pain Quotidien", "18 Place du Marche Saint-Honore", "Paris", "", "75001", "France"],
        ["s1_014", "HDFC Bank Ltd", "Senapati Bapat Marg Lower Parel", "Mumbai", "MH", "400013", "India"],
        ["s1_015", "Trader Joe's", "142 E 14th St", "New York", "NY", "10003", "United States"],
    ]

    train_s2 = [
        ["s2_101", "Starbucks", "102 Pike Street", "Seattle", "WA", "98101", "US"],
        ["s2_102", "Paul Boulangerie", "28 r de rivoli", "Paris", "Paris", "75004", "FR"],
        ["s2_103", "Infosys Ltd", "Hosur Rd Electronics City", "Bangalore", "Karnataka", "560100", "IN"],
        ["s2_104", "McDonalds", "110 North Carpenter Street", "Chicago", "IL", "60607", "United States"],
        ["s2_105", "TCS Ltd", "Nariman Point Nirmal Bldg", "Mumbai", "Maharashtra", "400021", "IND"],
        ["s2_106", "Carrefour Supermarket", "73 Blvd Saint Germain", "Paris", "Paris", "75005", "France"],
        ["s2_107", "Apple Retail Store", "1 Infinite Loop", "Cupertino", "CA", "95014", "USA"],
        ["s2_108", "Wipro Limited", "Sarjapur Rd Doddakannelli", "Bengaluru", "Karnataka", "560035", "India"],
        ["s2_109", "Chez Janou", "2 Rue Roger Verlomme", "Paris", "Paris", "75003", "FR"],
        ["s2_110", "Target Store", "1000 Nicollet Mall", "Minneapolis", "MN", "55403", "US"],
        ["s2_111", "Reliance Retail", "Nariman Pt Maker Chambers", "Mumbai", "MH", "400021", "India"],
        ["s2_112", "Subway", "200 Fifth Ave", "New York", "NY", "10010", "USA"],
        ["s2_113", "Pain Quotidien", "18 Pl du Marche St Honore", "Paris", "Paris", "75001", "FR"],
        ["s2_114", "HDFC Bank", "Lower Parel Senapati Bapat Marg", "Mumbai", "Maharashtra", "400013", "IN"],
        ["s2_115", "Trader Joes Grocery", "142 East 14th Street", "New York", "NY", "10003", "USA"],
        ["s2_999", "Seattle Coffee Works", "107 Pike St", "Seattle", "WA", "98101", "USA"],
    ]

    train_s3 = [
        ["s3_201", "Starbucks Coffee", "Pike St Suite 102", "Seattle", "WA", "98101", "USA"],
        ["s3_202", "Maison Paul Bakery", "28 Rue Rivoli", "Paris", "", "75004", "France"],
        ["s3_203", "Infosys Technologies", "Electronics City", "Bangalore", "KA", "560100", "India"],
        ["s3_204", "McDonald's Chicago HQ", "110 N Carpenter St", "Chicago", "IL", "60607", "US"],
        ["s3_205", "Tata Consultancy Services", "Nariman Point", "Mumbai", "MH", "400021", "India"],
        ["s3_206", "Supermarche Carrefour", "73 Bd Saint-Germain", "Paris", "Paris", "75005", "FR"],
        ["s3_207", "Apple Inc", "One Infinite Loop", "Cupertino", "CA", "95014", "US"],
        ["s3_208", "Wipro Infotech", "Sarjapur Road", "Bengaluru", "Karnataka", "560035", "India"],
        ["s3_209", "Restaurant Chez Janou", "2 R Roger Verlomme", "Paris", "Paris", "75003", "France"],
        ["s3_210", "Target", "1000 Nicollet Mall Ste 1", "Minneapolis", "MN", "55403", "United States"],
        ["s3_999", "Tata Motors Limited", "Bombay House Homi Mody Street", "Mumbai", "MH", "400001", "India"],
    ]

    train_gt = [
        ["s1_001", "s2_101,s3_201"],
        ["s1_002", "s2_102,s3_202"],
        ["s1_003", "s2_103,s3_203"],
        ["s1_004", "s2_104,s3_204"],
        ["s1_005", "s2_105,s3_205"],
        ["s1_006", "s2_106,s3_206"],
        ["s1_007", "s2_107,s3_207"],
        ["s1_008", "s2_108,s3_208"],
        ["s1_009", "s2_109,s3_209"],
        ["s1_010", "s2_110,s3_210"],
        ["s1_011", "s2_111"],
        ["s1_012", "s2_112"],
        ["s1_013", "s2_113"],
        ["s1_014", "s2_114"],
        ["s1_015", "s2_115"],
    ]

    test_s1 = [
        ["test_s1_01", "Amazon Web Services Inc", "410 Terry Ave N", "Seattle", "WA", "98109", "USA"],
        ["test_s1_02", "Galeries Lafayette Paris Haussmann", "40 Boulevard Haussmann", "Paris", "Paris", "75009", "France"],
        ["test_s1_03", "Zomato Limited", "Ground Floor 12A 94 Meghdoot", "New Delhi", "Delhi", "110019", "India"],
        ["test_s1_04", "Walmart Supercenter", "702 SW 8th St", "Bentonville", "AR", "72716", "United States"],
        ["test_s1_05", "Le Bristol Hotel Paris", "112 Rue du Faubourg Saint-Honore", "Paris", "", "75008", "France"],
        ["test_s1_06", "Swiggy Bundl Technologies", "Maruthi Chambers Outer Ring Rd", "Bengaluru", "Karnataka", "560068", "India"],
        ["test_s1_07", "Boutique Chanel Cambon", "31 Rue Cambon", "Paris", "Paris", "75001", "France"],
        ["test_s1_08", "Whole Foods Market", "525 N Lamar Blvd", "Austin", "TX", "78703", "US"],
        ["test_s1_09", "Mahindra and Mahindra Ltd", "Gateway Building Apollo Bunder", "Mumbai", "Maharashtra", "400001", "India"],
        ["test_s1_10", "Independent Bakery Without Matches", "99 Unknown Street", "Nowhere", "NW", "00000", "USA"],
    ]

    test_s2 = [
        ["test_s2_01", "AWS Amazon", "410 Terry Avenue North", "Seattle", "WA", "98109", "US"],
        ["test_s2_02", "Galeries Lafayette", "40 Bd Haussmann", "Paris", "Paris", "75009", "FR"],
        ["test_s2_03", "Zomato Ltd", "Meghdoot Bldg Nehru Place", "New Delhi", "Delhi", "110019", "IN"],
        ["test_s2_04", "Walmart Store", "702 Southwest 8th Street", "Bentonville", "AR", "72716", "USA"],
        ["test_s2_05", "Hotel Le Bristol", "112 R du Faubourg St-Honore", "Paris", "Paris", "75008", "France"],
        ["test_s2_06", "Swiggy HQ", "Outer Ring Road Maruthi Chambers", "Bangalore", "Karnataka", "560068", "India"],
        ["test_s2_07", "Chanel Cambon Store", "31 Rue Cambon", "Paris", "Paris", "75001", "FR"],
        ["test_s2_08", "Whole Foods Market Grocery", "525 North Lamar Boulevard", "Austin", "TX", "78703", "United States"],
        ["test_s2_09", "Mahindra & Mahindra", "Apollo Bunder Gateway Bldg", "Mumbai", "MH", "400001", "India"],
        ["test_s2_99", "Random Unrelated Shop", "123 Main Road", "Austin", "TX", "78701", "USA"],
    ]

    test_s3 = [
        ["test_s3_01", "Amazon AWS Headquarters", "410 Terry Ave", "Seattle", "WA", "98109", "United States"],
        ["test_s3_02", "Grand Magasin Galeries Lafayette", "40 Boulevard Haussmann", "Paris", "", "75009", "France"],
        ["test_s3_03", "Zomato Media Pvt Ltd", "94 Meghdoot Nehru Place", "New Delhi", "Delhi", "110019", "India"],
        ["test_s3_04", "Walmart Bentonville Store", "702 SW 8th St Ste 100", "Bentonville", "AR", "72716", "US"],
        ["test_s3_05", "Le Bristol Paris Oetker", "112 Rue Faubourg Saint Honore", "Paris", "Paris", "75008", "FR"],
        ["test_s3_06", "Bundl Technologies Swiggy", "Maruthi Chambers", "Bengaluru", "KA", "560068", "India"],
        ["test_s3_07", "Maison Chanel", "31 R Cambon", "Paris", "Paris", "75001", "France"],
        ["test_s3_08", "Whole Foods Austin HQ", "525 N Lamar", "Austin", "TX", "78703", "USA"],
        ["test_s3_09", "Mahindra Group", "Gateway Building", "Mumbai", "Maharashtra", "400001", "IND"],
        ["test_s3_99", "Unrelated Parisian Cafe", "10 Rue de la Paix", "Paris", "Paris", "75002", "France"],
    ]

    headers = ["id", "name", "address", "city", "state", "zip", "country"]
    gt_header = ["source1_id", "matched_id"]

    def write_tsv(filename: str, rows, hdr):
        with open(data_dir / filename, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            w.writerow(hdr)
            w.writerows(rows)

    write_tsv(TRAIN_SOURCE1, train_s1, headers)
    write_tsv(TRAIN_SOURCE2, train_s2, headers)
    write_tsv(TRAIN_SOURCE3, train_s3, headers)
    write_tsv(TRAIN_GROUND_TRUTH, train_gt, gt_header)

    write_tsv(TEST_SOURCE1, test_s1, headers)
    write_tsv(TEST_SOURCE2, test_s2, headers)
    write_tsv(TEST_SOURCE3, test_s3, headers)
    logger.info("Sample benchmark dataset created successfully.")


def run_pipeline(
    data_dir: Path = DATA_DIR,
    force_threshold: float = None,
    prefer_sklearn: bool = True
) -> None:
    logger.info("=" * 70)
    logger.info("AMAZON ML CHALLENGE - BUSINESS ENTITY RESOLUTION PIPELINE")
    logger.info("=" * 70)

    req_files = [TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3]
    missing = [f for f in req_files if not (data_dir / f).exists()]
    if missing:
        logger.warning(f"Missing dataset files in {data_dir}: {missing}")
        logger.info("Initializing complete benchmark dataset automatically...")
        create_sample_dataset(data_dir)

    # 1. Load Data
    logger.info("\n--- STEP 1: LOADING DATASETS ---")
    loader = DataLoader(data_dir)
    train_s1 = loader.load_source_file(TRAIN_SOURCE1, "s1")
    train_s2 = loader.load_source_file(TRAIN_SOURCE2, "s2")
    train_s3 = loader.load_source_file(TRAIN_SOURCE3, "s3")
    gt_map, positive_pairs = loader.load_ground_truth(TRAIN_GROUND_TRUTH)

    test_s1 = loader.load_source_file(TEST_SOURCE1, "s1")
    test_s2 = loader.load_source_file(TEST_SOURCE2, "s2")
    test_s3 = loader.load_source_file(TEST_SOURCE3, "s3")

    # 2. Preprocess & Normalize
    logger.info("\n--- STEP 2: PREPROCESSING & NORMALIZATION ---")
    train_s1_norm = Preprocessor.preprocess_dataframe(train_s1)
    train_s2_norm = Preprocessor.preprocess_dataframe(train_s2)
    train_s3_norm = Preprocessor.preprocess_dataframe(train_s3)

    test_s1_norm = Preprocessor.preprocess_dataframe(test_s1)
    test_s2_norm = Preprocessor.preprocess_dataframe(test_s2)
    test_s3_norm = Preprocessor.preprocess_dataframe(test_s3)

    train_target = pd.concat([train_s2_norm, train_s3_norm], ignore_index=True)
    test_target = pd.concat([test_s2_norm, test_s3_norm], ignore_index=True)

    logger.info(f"Train Pool: {len(train_s1_norm)} S1, {len(train_target)} Targets ({len(train_s2_norm)} S2 + {len(train_s3_norm)} S3)")
    logger.info(f"Test Pool:  {len(test_s1_norm)} S1, {len(test_target)} Targets ({len(test_s2_norm)} S2 + {len(test_s3_norm)} S3)")

    # 3. Fit TF-IDF Engine
    logger.info("\n--- STEP 3: INITIALIZING TF-IDF FEATURE ENGINE ---")
    all_texts = list(train_s1_norm["full_text"]) + list(train_target["full_text"])
    tfidf_engine = FastTfidfEngine()
    tfidf_engine.fit(all_texts)

    feature_extractor = FeatureExtractor(tfidf_engine)
    train_s1_records = train_s1_norm.to_dict(orient="records")
    train_target_records = train_target.to_dict(orient="records")

    s1_lookup = {r["id"]: r for r in train_s1_records}
    target_lookup = {r["id"]: r for r in train_target_records}

    feature_extractor.precompute_tfidf_cache(train_s1_records)
    feature_extractor.precompute_tfidf_cache(train_target_records)

    # 4. Blocking and Candidate Generation (Train)
    logger.info("\n--- STEP 4: BLOCKING & CANDIDATE GENERATION (TRAIN) ---")
    blocking_engine = BlockingEngine()
    blocking_engine.index_target_entities(train_target)
    train_pairs = blocking_engine.generate_candidate_pairs(
        train_s1_norm,
        positive_pairs=positive_pairs
    )

    # 5. Feature Matrix Construction
    logger.info("\n--- STEP 5: FEATURE EXTRACTION (TRAIN) ---")
    X_train_full = feature_extractor.build_feature_matrix(train_pairs, s1_lookup, target_lookup)
    y_train_full = np.array([1 if p in positive_pairs else 0 for p in train_pairs], dtype=np.int32)
    pos_count = int(np.sum(y_train_full == 1))
    neg_count = int(len(y_train_full) - pos_count)
    logger.info(f"Training feature matrix shape: {X_train_full.shape} | Positives: {pos_count}, Negatives: {neg_count}")

    # 6. Entity-aware Validation Split
    logger.info("\n--- STEP 6: GROUP VALIDATION SPLIT ---")
    all_train_s1_ids = list(train_s1_norm["id"])
    train_s1_ids, val_s1_ids = split_entities(all_train_s1_ids, val_ratio=VAL_SPLIT_RATIO)

    train_indices = [i for i, (s1_id, _) in enumerate(train_pairs) if s1_id in train_s1_ids]
    val_indices = [i for i, (s1_id, _) in enumerate(train_pairs) if s1_id in val_s1_ids]

    X_train_split, y_train_split = X_train_full[train_indices], y_train_full[train_indices]
    X_val_split, y_val_split = X_train_full[val_indices], y_train_full[val_indices]
    logger.info(f"Split sizes: Train pairs = {len(train_indices)}, Val pairs = {len(val_indices)}")

    # 7. Model Training & Threshold Tuning (F0.5)
    logger.info("\n--- STEP 7: VALIDATION TRAINING & F0.5 THRESHOLD TUNING ---")
    val_model = EntityResolutionModel(prefer_sklearn=prefer_sklearn)
    val_model.fit(X_train_split, y_train_split, feature_names=FeatureExtractor.FEATURE_NAMES)

    val_probs = val_model.predict_proba(X_val_split)[:, 1]
    tuned_thresh, best_metrics, history = tune_threshold(y_val_split, val_probs)

    final_threshold = force_threshold if force_threshold is not None else tuned_thresh
    logger.info(f"Selected Threshold for Deployment: {final_threshold:.2f}")

    validation_report = {
        "best_threshold": final_threshold,
        "validation_metrics": best_metrics,
        "history": history
    }
    save_metrics_report(validation_report, OUTPUT_METRICS)

    # 8. Train Final Model on All Training Data
    logger.info("\n--- STEP 8: FINAL MODEL RETRAINING ---")
    final_model = EntityResolutionModel(prefer_sklearn=prefer_sklearn)
    final_model.fit(X_train_full, y_train_full, feature_names=FeatureExtractor.FEATURE_NAMES)
    final_model.best_threshold = final_threshold
    final_model.save(MODEL_DIR)

    # 9. Test Inference & Generation of matching_results.tsv and candidate_pairs.tsv
    logger.info("\n--- STEP 9: TEST INFERENCE & TSV GENERATION ---")
    predictor = Predictor(
        model=final_model,
        feature_extractor=feature_extractor,
        blocking_engine=blocking_engine,
        threshold=final_threshold
    )
    predictor.predict_test_dataset(
        test_s1_df=test_s1_norm,
        test_target_df=test_target,
        gt_header=loader.ground_truth_columns,
        candidate_file=OUTPUT_CANDIDATE_PAIRS,
        results_file=OUTPUT_MATCHING_RESULTS,
    )

    # 10. Submission Validation
    logger.info("\n--- STEP 10: OFFICIAL VALIDATION AUDIT ---")
    expected_test_s1_ids = set(test_s1_norm["id"])
    valid_target_ids = set(test_target["id"])
    is_valid, issues = SubmissionValidator.validate(
        matching_file=OUTPUT_MATCHING_RESULTS,
        candidate_file=OUTPUT_CANDIDATE_PAIRS,
        expected_s1_ids=expected_test_s1_ids,
        valid_target_ids=valid_target_ids
    )

    if is_valid:
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Matching Results: {OUTPUT_MATCHING_RESULTS}")
        logger.info(f"Candidate Pairs:  {OUTPUT_CANDIDATE_PAIRS}")
        logger.info(f"Validation Report: {OUTPUT_METRICS}")
        logger.info("=" * 70)
    else:
        logger.error(f"Pipeline completed with {len(issues)} validation issues!")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge - Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to directory containing dataset TSV files")
    parser.add_argument("--threshold", type=float, default=None, help="Force a specific classification threshold (e.g. 0.65)")
    parser.add_argument("--create-sample-data", action="store_true", help="Generate benchmark sample data in dataset directory")
    parser.add_argument("--lookup", type=str, default=None, help="Check whether a business name has a similar match in the dataset")
    parser.add_argument("--interactive", action="store_true", help="Prompt for business names interactively")
    parser.add_argument("--pipeline", action="store_true", help="Run the full training and matching pipeline")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of similar businesses to show")
    return parser.parse_args()


def run_lookup_mode(search_name: str, data_dir: Path, limit: int, threshold: float | None = None):
    results = find_similar_businesses(search_name, data_dir, limit=limit, threshold=threshold)
    if not results:
        print(f"No business records were found in {data_dir}.")
        return

    print(f"\nBusiness lookup for: {search_name}")
    print("-" * 70)
    found_match = False
    for item in results:
        printable = "MATCH" if item["is_match"] else "NO MATCH"
        if item["is_match"]:
            found_match = True
        print(f"{printable:8} | {item['business_name']:<35} | source={item['source']:<8} | accuracy={item['score'] * 100:.1f}%")

    if found_match:
        print("\nResult: A similar business appears to exist in the dataset.")
    else:
        print("\nResult: No close match was found above the configured threshold.")


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data_dir)

    if args.lookup:
        run_lookup_mode(args.lookup, data_path, args.limit, args.threshold)
    elif args.interactive or not any([args.pipeline, args.create_sample_data, args.lookup]):
        while True:
            name = input("Enter a business name to check for a similar match (or press Enter to exit): ").strip()
            if not name:
                break
            run_lookup_mode(name, data_path, args.limit, args.threshold)
    elif args.pipeline:
        if args.create_sample_data:
            create_sample_dataset(data_path)
        run_pipeline(data_dir=data_path, force_threshold=args.threshold)
    else:
        if args.create_sample_data:
            create_sample_dataset(data_path)
        run_pipeline(data_dir=data_path, force_threshold=args.threshold)
