# src/test_set_b.py
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import time

# Inject absolute path parameters to prevent directory lookup loops
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import setup_logger, load_artifact
from src.features import extract_advanced_clinical_features
from src.evaluate import calculate_clinical_metrics, apply_stacking_ensemble
from src.model_dl import ClinicalLSTM
from src.data_pipeline import compile_raw_database, attach_outcomes

logger = setup_logger("set_b_test_orchestrator")

def process_and_evaluate_set_b():
    """Ingests Set B records, generates clean binary caches, and outputs a prospective metric scorecard."""
    logger.info("==============================================================")
    logger.info("       ICU FRAMEWORK: SET B PROSPECTIVE EVALUATION LOOP        ")
    logger.info("==============================================================")
    start_time = time.time()
    
    try:
        # --- STEP 1: LOAD ENSEMBLED ARTIFACT ARTIFACT BUNDLE ---
        logger.info("📂 Loading deep-frozen operational state dictionary from disk...")
        payload = load_artifact("hybrid_ensemble_core.joblib")
        
        lgbm_ensemble = payload["lgbm_fold_ensemble"]
        lstm_weights = payload["pytorch_lstm_state"]
        expected_features = payload["feature_names"]
        lstm_feature_dim = payload["lstm_feature_count"]
        seq_mean = payload["sequence_norm_mean"]
        seq_std = payload["sequence_norm_std"]
        stacker = payload["stacker"]
        stacker_input_order = payload["stacker_input_order"]

        # --- STEP 2: PARSE RAW TELEMETRY DATA VIA PARQUET RECONSTRUCTION ---
        raw_db_b = compile_raw_database(dataset_type="set-b")

        # --- STEP 3: EXECUTE TYPE-SAFE FEATURE TRANSFORMATIONS ---
        logger.info("⚡ Executing multi-engine temporal feature extractions...")
        feature_package = extract_advanced_clinical_features(raw_db_b, return_sequences=True)
        tabular_features = feature_package["tabular"]
        sequences_tensor = feature_package["sequences"]

        # --- STEP 4: ALIGN PROFILES WITH TRUTH OUTCOME LABEL REGISTRIES ---
        pre_merge_record_ids = tabular_features['RecordId'].values
        record_id_to_seq_row = {rid: i for i, rid in enumerate(pre_merge_record_ids)}

        master_dataset = attach_outcomes(tabular_features, dataset_type="set-b")
        y_true = master_dataset['In-hospital_death'].values

        # Missing columns stay NaN, not 0.0 -- LightGBM handles missing values natively
        for col in expected_features:
            if col not in master_dataset.columns:
                master_dataset[col] = np.nan

        X_tab_b = master_dataset[expected_features].copy()
        seq_b_raw = sequences_tensor[[record_id_to_seq_row[rid] for rid in master_dataset['RecordId'].values]]
        seq_b = (seq_b_raw - seq_mean) / seq_std

        logger.info(f"Set B matched cohort finalized. Total sample size: {len(y_true)} patients.")

        # --- STEP 5: RUN BLENDED HYBRID BATCH INFERENCE ---
        logger.info("🔮 Running inference across ensembled LightGBM decision trees...")
        test_lgbm_probs = np.mean([model.predict_proba(X_tab_b.values)[:, 1] for model in lgbm_ensemble], axis=0)

        logger.info("🔮 Running timeline inference through PyTorch LSTM...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lstm_skeleton = ClinicalLSTM(n_features=lstm_feature_dim).to(device)
        lstm_skeleton.load_state_dict(lstm_weights)
        lstm_skeleton.eval()

        with torch.no_grad():
            test_seq_tensor = torch.FloatTensor(seq_b).to(device)
            test_lstm_logits = lstm_skeleton(test_seq_tensor).cpu().numpy()
            test_lstm_probs = 1 / (1 + np.exp(-test_lstm_logits))
            test_lstm_probs = np.nan_to_num(test_lstm_probs, nan=0.142, posinf=1.0, neginf=0.0)

        test_probs = {"lgbm": test_lgbm_probs, "lstm": test_lstm_probs}
        final_ensemble_probs = apply_stacking_ensemble(stacker, stacker_input_order, test_probs)
        final_ensemble_probs = np.clip(final_ensemble_probs, 0.0, 1.0)
        
        # --- STEP 6: COMPILE OFFICIAL holdout METRIC SCORE CARD ---
        scores = calculate_clinical_metrics(y_true, final_ensemble_probs)
        
        logger.info("==================================================")
        logger.info("         OFFICIAL SET B PROSPECTIVE SCORE CARD     ")
        logger.info("==================================================")
        logger.info(f" Set B Area Under PR Curve (AUPRC):  {scores['AUPRC']:.4f}")
        logger.info(f" Set B Area Under ROC Curve (AUROC): {scores['AUROC']:.4f}")
        logger.info(f" Set B Balanced Event1 Metric Score: {scores['PhysioNet_Event1']:.4f}")
        logger.info(f" Set B Expected Brier Loss Matrix:   {scores['Brier_Loss']:.4f}")
        logger.info("==================================================")
        
        # --- STEP 7: EXPORT CLEAN PROCESSED DATASET LOGS FOR FUTURE TRAINING ---
        # Save a clean, wide dataset containing targets for our planned unified train loop
        master_dataset.to_csv(config.PROCESSED_DATA_DIR / "set_b_processed_features.csv", index=False)
        np.save(config.PROCESSED_DATA_DIR / "set_b_sequences.npy", seq_b)
        
        # Save output prediction audit log
        output_df = pd.DataFrame({
            "RecordId": master_dataset["RecordId"],
            "True_Label": y_true,
            "Ensemble_Risk_Score": final_ensemble_probs
        })
        output_path = config.PROCESSED_DATA_DIR / "set_b_prospective_audit.csv"
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"✨ Feature dataset logs and array configurations saved to: {config.PROCESSED_DATA_DIR}")
        logger.info(f"Total processing loop latency: {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        logger.critical(f"Set B prospective testing iteration failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    process_and_evaluate_set_b()