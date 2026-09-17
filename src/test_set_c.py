# src/test_set_c.py
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import setup_logger, load_artifact
from src.data_pipeline import compile_raw_database, attach_outcomes
from src.features import extract_advanced_clinical_features
from src.evaluate import calculate_clinical_metrics, apply_stacking_ensemble
from src.model_dl import ClinicalLSTM

logger = setup_logger("set_c_test_engine")

def evaluate_standalone_set_c():
    """
    Evaluates the frozen hybrid ensemble model strictly on Set C as an 
    isolated, unseen test holdout cohort (4,000 patients).
    """
    logger.info("==========================================================")
    logger.info("  STANDALONE EVALUATION: UNSEEN HOLDOUT SET C (4,000 PATIENTS)")
    logger.info("==========================================================")
    
    try:
        # 1. Load Trained Master Model Payload
        logger.info("[1/4] Ingesting frozen production artifact: hybrid_ensemble_core.joblib...")
        payload = load_artifact("hybrid_ensemble_core.joblib")
        
        lgbm_ensemble = payload["lgbm_fold_ensemble"]
        pytorch_state = payload["pytorch_lstm_state"]
        lstm_feature_dim = payload["lstm_feature_count"]
        seq_mean = payload["sequence_norm_mean"]
        seq_std = payload["sequence_norm_std"]
        stacker = payload["stacker"]
        stacker_input_order = payload["stacker_input_order"]

        # DYNAMIC FEATURE DISCOVERY: Interrogate LightGBM trees directly for the trained feature set
        if hasattr(lgbm_ensemble[0], "calibrated_classifiers_"):
            feature_checklist = list(lgbm_ensemble[0].calibrated_classifiers_[0].estimator.feature_name_)
        elif hasattr(lgbm_ensemble[0], "feature_name_"):
            feature_checklist = list(lgbm_ensemble[0].feature_name_)
        else:
            feature_checklist = payload["feature_names"]

        logger.info(f"[BOOT]: Verified model feature space. Dimensions required: {len(feature_checklist)}")

        # 2. Ingest and Extract Features Strictly for Set C
        logger.info("[2/4] Processing Set C records and building feature tensors...")
        db_c = compile_raw_database(dataset_type="set-c")
        pkg_c = extract_advanced_clinical_features(db_c, return_sequences=True)

        pre_merge_record_ids = pkg_c["tabular"]['RecordId'].values
        record_id_to_seq_row = {rid: i for i, rid in enumerate(pre_merge_record_ids)}

        master_c = attach_outcomes(pkg_c["tabular"], dataset_type="set-c")

        y_true_c = master_c['In-hospital_death'].values
        X_tabular_c = master_c.drop(columns=['In-hospital_death', 'RecordId'], errors='ignore')

        # Columns never seen in Set C stay NaN rather than 0.0
        X_aligned_c = pd.DataFrame(np.nan, index=np.arange(len(master_c)), columns=feature_checklist)
        for col in feature_checklist:
            if col in X_tabular_c.columns:
                X_aligned_c[col] = X_tabular_c[col].values

        seq_c_raw = pkg_c["sequences"][[record_id_to_seq_row[rid] for rid in master_c['RecordId'].values]]
        seq_c = (seq_c_raw - seq_mean) / seq_std

        # 3. Model Prediction Passes
        logger.info("[3/4] Running LightGBM Fold Ensemble & PyTorch BiLSTM inference pass...")

        # GBDT Probabilities across the trained feature checklist
        lgbm_probs = np.mean([model.predict_proba(X_aligned_c.values)[:, 1] for model in lgbm_ensemble], axis=0)

        # PyTorch BiLSTM Probabilities
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lstm_engine = ClinicalLSTM(n_features=lstm_feature_dim).to(device)
        lstm_engine.load_state_dict(pytorch_state)
        lstm_engine.eval()

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(seq_c).to(device)
            lstm_logits = lstm_engine(seq_tensor).cpu().numpy()
            lstm_probs = np.nan_to_num(1 / (1 + np.exp(-lstm_logits)), nan=0.142)

        # Stacked ensemble predictions using the learned LGBM/LSTM blend weights
        probs = {"lgbm": lgbm_probs, "lstm": lstm_probs}
        final_probs_c = np.clip(apply_stacking_ensemble(stacker, stacker_input_order, probs), 0.0, 1.0)
        
        # 4. Scorecard Calculation
        logger.info("[4/4] Calculating clinical performance metrics for Set C...")
        metrics_c = calculate_clinical_metrics(y_true_c, final_probs_c)
        
        print("\n" + "="*55)
        print("          UNSEEN SET C STANDALONE SCORECARD              ")
        print("="*55)
        print(f" Total Unseen Patients (Set C):   {len(y_true_c)}")
        print(f" Positives (In-Hospital Deaths):  {y_true_c.sum()} ({y_true_c.mean()*100:.2f}%)")
        print(f" AUROC Score:                     {metrics_c['AUROC']:.4f}")
        print(f" AUPRC Score:                     {metrics_c['AUPRC']:.4f}")
        print(f" PhysioNet Event 1 Score:         {metrics_c['PhysioNet_Event1']:.4f}")
        print(f" Brier Calibration Loss:          {metrics_c['Brier_Loss']:.4f}")
        print("="*55 + "\n")
        
    except Exception as e:
        logger.critical(f"Set C standalone evaluation stalled: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    evaluate_standalone_set_c()