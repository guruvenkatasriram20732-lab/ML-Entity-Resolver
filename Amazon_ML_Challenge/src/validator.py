"""
Submission validator module for Amazon ML Challenge - Business Entity Resolution.
Strictly verifies candidate pairs and final matching results for format conformity,
completeness, entity deduplication, and ID validity.
"""

import csv
import logging
from pathlib import Path
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

class SubmissionValidator:
    @staticmethod
    def validate(matching_file: Path, candidate_file: Path, expected_s1_ids: Set[str], valid_target_ids: Set[str]) -> Tuple[bool, List[str]]:
        issues = []
        if not candidate_file.exists() or candidate_file.stat().st_size == 0:
            issues.append(f"Candidate pairs file missing or empty: {candidate_file}")
        if not matching_file.exists() or matching_file.stat().st_size == 0:
            issues.append(f"Matching results file missing or empty: {matching_file}")
            return False, issues

        seen_s1 = set()
        with open(matching_file, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if not row:
                    continue
                s1_id = row[0].strip()
                match_val = row[1].strip() if len(row) > 1 else ""
                if s1_id in seen_s1:
                    issues.append(f"Duplicate Source 1 ID: {s1_id}")
                seen_s1.add(s1_id)
                if match_val:
                    for m in match_val.split(","):
                        m = m.strip()
                        if m not in valid_target_ids:
                            issues.append(f"Invalid target ID {m} for {s1_id}")
                        if m == s1_id:
                            issues.append(f"Self-match detected: {s1_id}")

        if expected_s1_ids - seen_s1:
            issues.append(f"Missing {len(expected_s1_ids - seen_s1)} test Source 1 entities")
        valid = (len(issues) == 0)
        logger.info(">>> VALIDATION PASSED <<<" if valid else f">>> VALIDATION FAILED: {len(issues)} issues <<<")
        return valid, issues
