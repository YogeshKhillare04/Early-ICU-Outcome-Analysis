# src/train.py
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import config
from src.utils import setup_logger, enforce_reproducibility, save_artifact
from src.data_pipeline import compile_raw_database, attach_outcomes
from src.features import extract_advanced_clinical_features
from src.evaluate import run_stratified_validation, evaluate_ensemble, apply_stacking_ensemble, calculate_clinical_metrics
from src.model_dl import train_lstm

logger = setup_logger("pipeline_orchestrator")

def execute_end_to_end_training_pipeline():
    """Coordinates data assembly, feature extraction, cross-validation modeling, and ensembling."""
    start_time = datetime.now()
    logger.info("==============================================================")
    logger.info("   EARLY ICU OUTCOME ANALYSIS PIPELINE INITIALIZED      ")
    logger.info("==============================================================")
    
    try:
        enforce_reproducibility(seed=config.SEED)
        
        # Load the raw patient files.
        logger.info("Loading and compiling patient files...")
        raw_db = compile_raw_database()
        
        # Build both the summary features and the hourly sequences.
        logger.info("Building patient features...")
        feature_package = extract_advanced_clinical_features(raw_db, return_sequences=True)
        tabular_features = feature_package["tabular"]
        sequences_tensor = feature_package["sequences"] # Shape: (Patients, 48 Hours, Features)

        # Add the outcome labels and keep the sequence rows aligned with them.
        logger.info("Attaching outcome labels...")
        # attach_outcomes merges and can reorder rows, so re-gather sequences by RecordId
        # afterwards rather than assuming they still line up positionally
        pre_merge_record_ids = tabular_features['RecordId'].values
        record_id_to_seq_row = {rid: i for i, rid in enumerate(pre_merge_record_ids)}

        master_dataset = attach_outcomes(tabular_features)
        sequences_tensor = sequences_tensor[[record_id_to_seq_row[rid] for rid in master_dataset['RecordId'].values]]

        record_ids = master_dataset['RecordId'].values if 'RecordId' in master_dataset.columns else np.arange(len(master_dataset))
        y = master_dataset['In-hospital_death'].values
        
        # Filter down feature dataframes to keep them pure before passing to estimators
        X_tabular = master_dataset.drop(columns=['In-hospital_death', 'RecordId'], errors='ignore')
        
        # Keep a final holdout set that is not used during training.
        logger.info("Splitting development and holdout data...")
        indices = np.arange(len(y))
        
        # Split all indices to keep our tabular tables and sequence arrays perfectly aligned
        idx_dev, idx_test, y_dev, y_test = train_test_split(
            indices, y, test_size=0.15, stratify=y, random_state=config.SEED
        )
        
        # Sub-slice data arrays using our split indexes
        X_tab_dev, X_tab_test = X_tabular.iloc[idx_dev].reset_index(drop=True), X_tabular.iloc[idx_test].reset_index(drop=True)
        seq_dev, seq_test = sequences_tensor[idx_dev], sequences_tensor[idx_test]
        record_ids_test = record_ids[idx_test]

        # Normalize using dev-split stats only, then apply the same transform to the test split
        seq_mean = seq_dev.mean(axis=(0, 1))
        seq_std = seq_dev.std(axis=(0, 1))
        seq_std[seq_std < 1e-6] = 1.0
        seq_dev_norm = (seq_dev - seq_mean) / seq_std
        seq_test_norm = (seq_test - seq_mean) / seq_std

        logger.info("Training the LightGBM models...")
        lgbm_fold_models, oof_lgbm_probs, cv_lgbm_scores = run_stratified_validation(X_tab_dev, y_dev)

        logger.info("Training the LSTM model...")
        lstm_model, oof_lstm_probs = train_lstm(seq_dev_norm, y_dev, n_epochs=25, batch_size=64)

        logger.info("Combining the out-of-fold predictions...")
        oof_probs = {"lgbm": oof_lgbm_probs, "lstm": oof_lstm_probs}
        cv_ensemble_package = evaluate_ensemble(oof_probs, y_dev)
        stacker = cv_ensemble_package["stacker"]
        stacker_input_order = cv_ensemble_package["stacker_input_order"]

        logger.info("Evaluating the ensemble on the holdout data...")

        # 1. Generate and average predictions across all 5 LightGBM folds
        test_lgbm_probs = np.mean([model.predict_proba(X_tab_test.values)[:, 1] for model in lgbm_fold_models], axis=0)

        # Generate predictions from the LSTM.
        import torch
        lstm_model.eval()
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            test_seq_tensor = torch.FloatTensor(seq_test_norm).to(device)
            test_lstm_logits = lstm_model(test_seq_tensor).cpu().numpy()

            test_lstm_probs = 1 / (1 + np.exp(-test_lstm_logits))
            test_lstm_probs = np.nan_to_num(test_lstm_probs, nan=0.0, posinf=1.0, neginf=0.0)

        # Combine predictions using the learned stacker.
        test_probs = {"lgbm": test_lgbm_probs, "lstm": test_lstm_probs}
        final_test_ensemble_probs = apply_stacking_ensemble(stacker, stacker_input_order, test_probs)

        final_test_ensemble_probs = np.clip(final_test_ensemble_probs, 0.0, 1.0)
        final_test_scores = calculate_clinical_metrics(y_test, final_test_ensemble_probs)
        logger.info("==================================================")
        logger.info("       FINAL HOLDOUT PERFORMANCE       ")
        logger.info("==================================================")
        logger.info(f" Test Set Area Under PR Curve (AUPRC): {final_test_scores['AUPRC']:.4f}")
        logger.info(f" Test Set Area Under ROC Curve (AUROC): {final_test_scores['AUROC']:.4f}")
        logger.info(f" Test Balanced Event1 Score:            {final_test_scores['PhysioNet_Event1']:.4f}")
        logger.info(f" Test Calibration Event2 Score:         {final_test_scores['Brier_Loss']:.4f}")
        logger.info("==================================================")
        
        logger.info(f"Saving the trained models to {config.MODEL_DIR}...")
        
        production_payload = {
            "lgbm_fold_ensemble": lgbm_fold_models,
            "pytorch_lstm_state": lstm_model.state_dict(),
            "lstm_feature_count": seq_dev.shape[2],
            "sequence_feature_names": config.SEQUENCE_FEATURES,
            "sequence_norm_mean": seq_mean,
            "sequence_norm_std": seq_std,
            "stacker": stacker,
            "stacker_input_order": stacker_input_order,
            "feature_names": list(X_tabular.columns),
            "historical_test_scores": final_test_scores
        }
        save_artifact(production_payload, "hybrid_ensemble_core.joblib")

        # Keep the holdout predictions for later inspection.
        test_audit_df = pd.DataFrame({
            "RecordId": record_ids_test,
            "True_Label": y_test,
            "LGBM_Risk_Score": test_lgbm_probs,
            "LSTM_Risk_Score": test_lstm_probs,
            "Ensemble_Calibrated_Score": final_test_ensemble_probs
        })
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        test_audit_df.to_csv(config.PROCESSED_DATA_DIR / "final_holdout_test_audit.csv", index=False)
        
        logger.info(f"Processing execution complete. Elapsed time: {datetime.now() - start_time}")
        
    except Exception as e:
        logger.critical(f"System processing stopped by runtime exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    execute_end_to_end_training_pipeline()