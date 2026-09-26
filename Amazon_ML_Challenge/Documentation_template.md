# Technical Report: Business Entity Resolution Pipeline

## 1. Executive Summary
This document outlines the architecture, candidate blocking methodology, feature engineering pipeline, machine learning model, and precision-prioritized threshold tuning strategy developed for the Amazon ML Challenge – Business Entity Resolution.

## 2. Preprocessing & Normalization
- **Unicode Transliteration**: Applied NFKD decomposition to convert accented characters (e.g., `é`, `ü`, `ç`) into standard ASCII characters.
- **Corporate Entity Normalization**: Stripped and standardized legal business suffixes (`Inc`, `LLC`, `Corp`, `Ltd`, `Pvt Ltd`, `GmbH`, `SARL`, `SA`).
- **Address Expansion**: Standardized abbreviations (`St` $\to$ `Street`, `Ave` $\to$ `Avenue`, `Blvd` $\to$ `Boulevard`, `Rd` $\to$ `Road`, `Ste` $\to$ `Suite`).
- **Country Standardization**: Dynamic mapping to ISO-2 codes (`USA` / `United States` $\to$ `US`, `India` $\to$ `IN`, `France` $\to$ `FR`) without hard-coding specific country lists.

## 3. Scalable Candidate Blocking
To prevent memory blowups on ~1GB datasets:
- Avoided naive Cartesian joins ($O(N_1 \times N_{2+3})$).
- Created multi-key inverted index tables on:
  1. `Country + 3-char Name Prefix`
  2. `Country + First Significant Token`
  3. `Country + Postal Code`
  4. `Country + Longest Token`
- Heavy block pruning for stop tokens with $> 1500$ occurrences.
- Top-K candidate capping ($K \le 60$) using lexical Jaccard pre-scoring.

## 4. Feature Engineering (21 Features)
1. `name_levenshtein_sim`: Character-level edit similarity (0 to 1).
2. `name_jaro_winkler`: Prefix-weighted string distance.
3. `name_token_jaccard`: Word token overlap.
4. `name_token_containment`: Subset token containment.
5. `name_char_3gram_jaccard`: Sub-word character trigram Jaccard.
6. `name_exact_match`: Exact normalized string equality.
7. `name_prefix_match`: 3-character prefix matching.
8. `name_first_token_match`: First word match.
9. `addr_levenshtein_sim`: Street address edit distance similarity.
10. `addr_token_jaccard`: Address word token Jaccard.
11. `addr_number_match`: Street numeric digits comparison (+1 match, -1 conflict, 0 missing).
12. `city_match`: Exact city equality.
13. `state_match`: State/region equality.
14. `zip_match`: Postal code equality.
15. `country_match`: Standardized ISO country match.
16. `tfidf_cosine_sim`: Sublinear character 3-gram TF-IDF cosine similarity.
17. `len_diff`: Absolute character length difference.
18. `len_ratio`: Length ratio min/max.
19. `token_diff`: Absolute token count difference.
20. `is_source2`: Binary source indicator.
21. `is_source3`: Binary source indicator.

## 5. Machine Learning Classifier & Threshold Tuning
- **Model**: Regularized Logistic Regression with class-imbalance weighting.
- **Validation**: Group split (80% train, 20% validation) grouped strictly on Source 1 entity IDs.
- **Metric**: F0.5 Score ($F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$).
- **Threshold Optimization**: Swept classification probability cutoffs from $0.20$ to $0.95$ in $0.05$ increments to locate the global maximum $F_{0.5}$.
