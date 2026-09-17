# src/tune_lgbm.py
"""Randomized hyperparameter search for the LightGBM base learner. Only touches the
development split, never the final holdout. Winning config gets copied into config.LGBM_PARAMS."""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import setup_logger, enforce_reproducibility
from src.data_pipeline import compile_raw_database, attach_outcomes
from src.features import extract_advanced_clinical_features
from src.evaluate import run_stratified_validation

logger = setup_logger("lgbm_tuning_engine")

SEARCH_SPACE = {
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 4, 5, 6, -1],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08],
    'min_child_samples': [5, 10, 20, 40],
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'lambda_l1': [0.0, 0.1, 0.5, 1.0],
    'lambda_l2': [0.0, 0.1, 0.5, 1.0, 5.0],
    'scale_pos_weight': [4.0, 5.0, 6.0, 7.0, 8.0],
}

N_TRIALS = 40


def sample_config(rng: np.random.RandomState) -> dict:
    return {k: rng.choice(v).item() if hasattr(rng.choice(v), "item") else rng.choice(v) for k, v in SEARCH_SPACE.items()}


def run_search():
    enforce_reproducibility(seed=config.SEED)
    logger.info("Loading Set A and building feature matrix for hyperparameter search...")
    raw_db = compile_raw_database(dataset_type="set-a")
    feature_package = extract_advanced_clinical_features(raw_db, return_sequences=False)
    master_dataset = attach_outcomes(feature_package["tabular"])

    y = master_dataset['In-hospital_death'].values
    X_tabular = master_dataset.drop(columns=['In-hospital_death', 'RecordId'], errors='ignore')

    indices = np.arange(len(y))
    idx_dev, _, y_dev, _ = train_test_split(
        indices, y, test_size=config.TEST_SIZE_PROPORTION, stratify=y, random_state=config.SEED
    )
    X_tab_dev = X_tabular.iloc[idx_dev].reset_index(drop=True)

    rng = np.random.RandomState(config.SEED)
    results = []
    base_static = {k: v for k, v in config.LGBM_PARAMS.items()
                    if k not in SEARCH_SPACE and k not in ('n_estimators',)}

    for trial in range(N_TRIALS):
        trial_params = sample_config(rng)
        candidate = {**base_static, **trial_params}

        original = dict(config.LGBM_PARAMS)
        config.LGBM_PARAMS.clear()
        config.LGBM_PARAMS.update(candidate)
        try:
            _, _, metrics = run_stratified_validation(X_tab_dev, y_dev)
            results.append({"params": trial_params, "AUPRC": metrics["AUPRC"], "AUROC": metrics["AUROC"],
                             "Event1": metrics["PhysioNet_Event1"]})
            logger.info(f"Trial {trial+1}/{N_TRIALS} -> AUPRC: {metrics['AUPRC']:.4f} | params: {trial_params}")
        except Exception as e:
            logger.warning(f"Trial {trial+1} failed: {e}")
        finally:
            config.LGBM_PARAMS.clear()
            config.LGBM_PARAMS.update(original)

    results.sort(key=lambda r: r["AUPRC"], reverse=True)
    logger.info("=" * 60)
    logger.info("TOP 5 CONFIGS BY OOF AUPRC:")
    for r in results[:5]:
        logger.info(f"  AUPRC={r['AUPRC']:.4f} AUROC={r['AUROC']:.4f} Event1={r['Event1']:.4f} | {r['params']}")
    logger.info("=" * 60)

    out_path = config.PROCESSED_DATA_DIR / "lgbm_tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_search()
