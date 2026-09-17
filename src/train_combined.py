# src/train_combined.py
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import setup_logger, enforce_reproducibility, save_artifact
from src.data_pipeline import compile_raw_database, attach_outcomes
from src.features import extract_advanced_clinical_features
from src.evaluate import run_stratified_validation, evaluate_ensemble, apply_stacking_ensemble, calculate_clinical_metrics
from src.model_dl import train_lstm

logger = setup_logger("combined_master_train_engine")

def execute_combined_master_training():
    """
    Unified Master Pipeline.
    Processes Set A and Set B feature spaces independently before stacking matrix boundaries,
    guaranteeing perfect index alignment between tabular data rows and PyTorch sequence cubes.
    """
    start_time = datetime.now()
    logger.info("==============================================================")
    logger.info("  ICU MORTALITY SYSTEM: UNIFIED MASTER RETRAINING PIPELINE     ")
    logger.info("==============================================================")
    
    try:
        enforce_reproducibility(seed=config.SEED)
        
        # --- PHASE 1: COMPRESSED BINARY ETL INGESTION ---
        logger.info("[ETL]: Ingesting raw databases from compressed Parquet caching pools...")
        db_a = compile_raw_database(dataset_type="set-a")
        db_b = compile_raw_database(dataset_type="set-b")
        
        # --- PHASE 2: DECOUPLED ADVANCED FEATURE EXTRACTION ---
        # FIXED: Extract features and sequences separately to prevent coordinate mixing errors
        logger.info("[FEATURES]: Extracting structural profiles from Set A pools...")
        pkg_a = extract_advanced_clinical_features(db_a, return_sequences=True)
        
        logger.info("[FEATURES]: Extracting structural profiles from Set B pools...")
        pkg_b = extract_advanced_clinical_features(db_b, return_sequences=True)
        
        # --- PHASE 3: GROUND TRUTH OUTCOME TARGETING & ALIGNED STACKING ---
        logger.info("[REGISTRY]: Mapping independent outcome survival labels...")
        master_a = attach_outcomes(pkg_a["tabular"], dataset_type="set-a")
        master_b = attach_outcomes(pkg_b["tabular"], dataset_type="set-b")

        # Concatenate tabular arrays vertically, maintaining chronological sequence ordering
        master_dataset = pd.concat([master_a, master_b], ignore_index=True)\
                           .drop_duplicates(subset=['RecordId'])\
                           .reset_index(drop=True)

        # Re-gather sequences by RecordId rather than positional concat, since the merge above
        # can reorder or drop rows
        seq_a = pkg_a["sequences"]
        seq_b = pkg_b["sequences"]
        record_id_to_seq_row = {}
        for i, rid in enumerate(pkg_a["tabular"]['RecordId'].values):
            record_id_to_seq_row[rid] = ('a', i)
        for i, rid in enumerate(pkg_b["tabular"]['RecordId'].values):
            record_id_to_seq_row.setdefault(rid, ('b', i))

        seq_rows = []
        for rid in master_dataset['RecordId'].values:
            source, row_idx = record_id_to_seq_row[rid]
            seq_rows.append(seq_a[row_idx] if source == 'a' else seq_b[row_idx])
        sequences_cube = np.stack(seq_rows, axis=0)

        y_master = master_dataset['In-hospital_death'].values
        X_tabular = master_dataset.drop(columns=['In-hospital_death', 'RecordId'], errors='ignore')
        
        logger.info(f"[REGISTRY]: Index matching validated. Wide matrix dimension: {X_tabular.shape} across {len(y_master)} patients.")
        
        # --- PHASE 4: STRATIFIED DEV/TEST HOLDOUT ---
        indices = np.arange(len(y_master))
        
        # INLINE FALLBACK GATE: Safeguards against config naming attribute discrepancies
        test_size_val = getattr(config, "TEST_SIZE_PROPORTION", 0.15)
        
        idx_dev, idx_test, y_dev, y_test = train_test_split(
            indices, 
            y_master, 
            test_size=test_size_val, 
            stratify=y_master, 
            random_state=config.SEED
        )
        
        X_tab_dev = X_tabular.iloc[idx_dev].reset_index(drop=True)
        X_tab_test = X_tabular.iloc[idx_test].reset_index(drop=True)

        # Aligned tensor slices can now be safely passed to the PyTorch LSTM layers
        seq_dev = sequences_cube[idx_dev]
        seq_test = sequences_cube[idx_test]

        # Normalize sequence channels using only the development split's statistics
        seq_mean = seq_dev.mean(axis=(0, 1))
        seq_std = seq_dev.std(axis=(0, 1))
        seq_std[seq_std < 1e-6] = 1.0
        seq_dev_norm = (seq_dev - seq_mean) / seq_std
        seq_test_norm = (seq_test - seq_mean) / seq_std

        # --- PHASE 5: GRADIENT BOOSTING ENSEMBLE TRAINING ---
        logger.info("[MODELING STEP 1]: Fit 5-Fold Stratified LightGBM Estimators...")
        lgbm_fold_ensemble, oof_lgbm_probs, _ = run_stratified_validation(X_tab_dev, y_dev)
        expected_features = list(X_tab_dev.columns)

        # --- PHASE 6: DEEP LONGITUDINAL BIDIRECTIONAL RECURRENT NETWORKS ---
        logger.info("[MODELING STEP 2]: Training PyTorch Bidirectional LSTM on balanced cohorts...")
        lstm_champion_model, oof_lstm_probs = train_lstm(seq_dev_norm, y_dev, n_epochs=25, batch_size=64)

        # --- PHASE 7: HYBRID ENSEMBLE PREDICTIONS BLENDING ---
        logger.info("[EVALUATION]: Learning stacked ensemble weights from out-of-fold predictions...")
        oof_probs = {"lgbm": oof_lgbm_probs, "lstm": oof_lstm_probs}
        cv_ensemble_package = evaluate_ensemble(oof_probs, y_dev)
        stacker = cv_ensemble_package["stacker"]
        stacker_input_order = cv_ensemble_package["stacker_input_order"]

        logger.info("[EVALUATION]: Testing ensembled weights over the locked 15% validation holdout...")
        test_lgbm_probs = np.mean([model.predict_proba(X_tab_test.values)[:, 1] for model in lgbm_fold_ensemble], axis=0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        lstm_champion_model.eval()
        with torch.no_grad():
            test_seq_tensor = torch.FloatTensor(seq_test_norm).to(device)
            test_lstm_logits = lstm_champion_model(test_seq_tensor).cpu().numpy()
            test_lstm_probs = 1 / (1 + np.exp(-test_lstm_logits))
            test_lstm_probs = np.nan_to_num(test_lstm_probs, nan=0.142, posinf=1.0, neginf=0.0)

        test_probs = {"lgbm": test_lgbm_probs, "lstm": test_lstm_probs}
        final_test_ensemble_probs = apply_stacking_ensemble(stacker, stacker_input_order, test_probs)
        final_test_ensemble_probs = np.clip(final_test_ensemble_probs, 0.0, 1.0)

        final_test_scores = calculate_clinical_metrics(y_test, final_test_ensemble_probs)
        
        # FIXED: Core metric log string typos corrected to reflect true target tracking variables
        logger.info("==================================================")
        logger.info("      UPGRADED MASTER HYBRID MODEL TEST SCORES     ")
        logger.info("==================================================")
        logger.info(f" Combined Master Cohort AUROC:        {final_test_scores['AUROC']:.4f}")
        logger.info(f" Combined Master Cohort AUPRC:        {final_test_scores['AUPRC']:.4f}")
        logger.info(f" Combined Master Cohort Event1:       {final_test_scores['PhysioNet_Event1']:.4f}")
        logger.info(f" Combined Master Cohort Brier Loss:   {final_test_scores['Brier_Loss']:.4f}")
        logger.info("==================================================")
        
        # --- PHASE 8: COMPRESSED JOBLIB SERIALIZATION ---
        # FIXED: Cleaned up notation spelling parameters
        logger.info("[SERIALIZATION]: Packing optimized model dependencies to single compressed Joblib payload...")
        production_bundle = {
            "lgbm_fold_ensemble": lgbm_fold_ensemble,
            "pytorch_lstm_state": lstm_champion_model.state_dict(),
            "lstm_feature_count": seq_dev.shape[2],
            "sequence_feature_names": config.SEQUENCE_FEATURES,
            "sequence_norm_mean": seq_mean,
            "sequence_norm_std": seq_std,
            "stacker": stacker,
            "stacker_input_order": stacker_input_order,
            "feature_names": expected_features,
            "historical_test_scores": final_test_scores
        }
        save_artifact(production_bundle, "hybrid_ensemble_core.joblib")
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"✨ Master training sequence completed perfectly. Elapsed Runtime: {elapsed_time}")
        
    except Exception as e:
        logger.critical(f"Unified master training pipeline execution stalled: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    execute_combined_master_training()