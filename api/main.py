# api/main.py
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import setup_logger, load_artifact
from src.model_dl import ClinicalLSTM
from src.evaluate import apply_stacking_ensemble
from src.features import construct_physiological_ratios, TRAJECTORY_VITALS

api_logger = setup_logger("api_serving_engine")

app = FastAPI(
    title="Early ICU Outcome Analysis API",
    version="2.7.0",
    description="API for generating ICU mortality estimates and displaying the main model features behind each estimate."
)

# Loaded when the API starts.
MODEL_PAYLOAD = None
LGBM_ENSEMBLE = None
PYTORCH_STATE = None
FEATURE_CHECKLIST = []
SHAP_EXPLAINER = None
STACKER = None
STACKER_INPUT_ORDER = []
SEQUENCE_FEATURE_NAMES = []
SEQ_NORM_MEAN = None
SEQ_NORM_STD = None

@app.on_event("startup")
def load_model_artifacts():
    global MODEL_PAYLOAD, LGBM_ENSEMBLE, PYTORCH_STATE, FEATURE_CHECKLIST, SHAP_EXPLAINER
    global STACKER, STACKER_INPUT_ORDER, SEQUENCE_FEATURE_NAMES, SEQ_NORM_MEAN, SEQ_NORM_STD
    try:
        api_logger.info("Loading model artifact bundle from disk...")
        MODEL_PAYLOAD = load_artifact("hybrid_ensemble_core.joblib")
        LGBM_ENSEMBLE = MODEL_PAYLOAD["lgbm_fold_ensemble"]
        PYTORCH_STATE = MODEL_PAYLOAD["pytorch_lstm_state"]
        STACKER = MODEL_PAYLOAD["stacker"]
        STACKER_INPUT_ORDER = MODEL_PAYLOAD["stacker_input_order"]
        SEQUENCE_FEATURE_NAMES = MODEL_PAYLOAD["sequence_feature_names"]
        SEQ_NORM_MEAN = MODEL_PAYLOAD["sequence_norm_mean"]
        SEQ_NORM_STD = MODEL_PAYLOAD["sequence_norm_std"]

        # Read the feature order from the fitted model.
        if hasattr(LGBM_ENSEMBLE[0], "calibrated_classifiers_"):
            FEATURE_CHECKLIST = list(LGBM_ENSEMBLE[0].calibrated_classifiers_[0].estimator.feature_name_)
            base_estimator = LGBM_ENSEMBLE[0].calibrated_classifiers_[0].estimator
        elif hasattr(LGBM_ENSEMBLE[0], "feature_name_"):
            FEATURE_CHECKLIST = list(LGBM_ENSEMBLE[0].feature_name_)
            base_estimator = LGBM_ENSEMBLE[0]
        else:
            FEATURE_CHECKLIST = MODEL_PAYLOAD.get("feature_names", [])
            base_estimator = LGBM_ENSEMBLE[0]

        # SHAP is used to show the strongest feature contributions.
        SHAP_EXPLAINER = shap.TreeExplainer(base_estimator)
        api_logger.info(f"Model artifact loaded successfully. Feature dimensions: {len(FEATURE_CHECKLIST)}")

    except Exception as e:
        api_logger.critical(f"Failed to load model artifacts: {str(e)}", exc_info=True)
        raise RuntimeError(f"Startup Failure: {str(e)}")

# --- PYDANTIC CONTRACT SCHEMAS ---
class ObservationItem(BaseModel):
    Parameter: str
    Value: float

class PatientPayload(BaseModel):
    Age: float = Field(..., ge=15.0, le=110.0, description="Patient age in years")
    Gender: int = Field(..., ge=0, le=1, description="Gender indicator (0: Female, 1: Male)")
    Observations: List[ObservationItem] = Field(..., description="List of recorded clinical vitals/labs")

class RankedPatientPayload(BaseModel):
    Patient_ID: str = Field(..., min_length=1, description="Portfolio or hospital-system patient identifier")
    Patient: PatientPayload

# Names used when displaying model features in the API response.
CLINICAL_NAME_MAP = {
    "Age": "Patient Age",
    "Gender": "Biological Sex",
    "Height": "Patient Height",
    "Weight": "Patient Weight",
    "ICUType": "ICU Ward Type",
    "HR": "Heart Rate",
    "GCS": "Glasgow Coma Scale (GCS)",
    "SysABP": "Systolic Blood Pressure (Invasive)",
    "NISysABP": "Systolic Blood Pressure (Non-Invasive)",
    "DiasABP": "Diastolic Blood Pressure (Invasive)",
    "NIDiasABP": "Diastolic Blood Pressure (Non-Invasive)",
    "MAP": "Mean Arterial Pressure (MAP)",
    "NIMAP": "Mean Arterial Pressure (Non-Invasive)",
    "Temp": "Core Body Temperature",
    "RespRate": "Respiration Rate",
    "Urine": "Urine Output",
    "BUN": "Blood Urea Nitrogen (BUN)",
    "Creatinine": "Serum Creatinine",
    "Platelets": "Platelet Count",
    "WBC": "White Blood Cell Count",
    "Glucose": "Serum Glucose",
    "FiO2": "Inspired Oxygen Fraction (FiO2)",
    "pH": "Blood pH Balance",
    "PaO2": "Partial Pressure of Oxygen (PaO2)",
    "PaCO2": "Partial Pressure of Carbon Dioxide (PaCO2)",
    "HCO3": "Serum Bicarbonate (HCO3)",
    "Lactate": "Serum Lactate",
    "Mg": "Serum Magnesium",
    "K": "Serum Potassium",
    "Na": "Serum Sodium",
    "Albumin": "Serum Albumin",
    "Bilirubin": "Serum Bilirubin",
    "ALP": "Alkaline Phosphatase",
    "ALT": "Alanine Aminotransferase (ALT)",
    "AST": "Aspartate Aminotransferase (AST)",
    "HCT": "Hematocrit",
    "SaO2": "Oxygen Saturation (SaO2)",
    "Shock_Index": "Shock Index (HR / Systolic BP)",
    "RPP": "Rate Pressure Product",
    "PF_Ratio": "P/F Ratio (Lung Function)",
    "BUN_Creat_Ratio": "BUN-to-Creatinine Ratio",
    "GCS_Trend_Severity": "GCS Deterioration Severity",
    "Total_Instability_Score": "Overall Physiological Instability Score",
}

def format_feature_label(raw_name: str) -> str:
    """Maps raw feature tokens to professional clinical terminology."""
    if raw_name in CLINICAL_NAME_MAP:
        return CLINICAL_NAME_MAP[raw_name]

    # Columns are "{param_code}_{suffix}" (e.g. "HR_mean") -- look up the base code exactly
    if '_' in raw_name:
        base, _, suffix = raw_name.partition('_')
        if base in CLINICAL_NAME_MAP:
            clean = f"{CLINICAL_NAME_MAP[base]}_{suffix}"
        else:
            clean = raw_name
    else:
        clean = raw_name

    clean = (
        clean.replace("_missing", " (Never Measured Marker)")
             .replace("_delta", " (48h Trend Change)")
             .replace("_", " ")
    )
    return clean.title()

def build_review_suggestions(payload: PatientPayload, risk_probability: float) -> List[str]:
    values = {item.Parameter: item.Value for item in payload.Observations}
    suggestions = []

    if risk_probability >= 0.50:
        suggestions.append("Prioritize this record for clinician review because the predicted risk is high.")
    elif risk_probability >= 0.20:
        suggestions.append("Include this record in the moderate-risk review queue and monitor new observations.")
    else:
        suggestions.append("Keep this record in routine monitoring; the model currently estimates lower risk.")

    if values.get("GCS", 15) < 9:
        suggestions.append("Review the low GCS observation and confirm that it is recorded correctly.")
    if values.get("SysBP", values.get("SysABP", 120)) < 90:
        suggestions.append("Review the low systolic blood pressure observation promptly.")
    if values.get("HR", 80) > 120:
        suggestions.append("Review the elevated heart-rate observation and its recent trend.")
    if values.get("Temp", 37) >= 38.5:
        suggestions.append("Review the elevated temperature observation alongside other clinical findings.")

    if len(values) < 5:
        suggestions.append("Consider whether additional available observations should be recorded before reassessment.")

    return suggestions

@app.post("/predict")
def predict_mortality(payload: PatientPayload):
    if LGBM_ENSEMBLE is None or SHAP_EXPLAINER is None:
        raise HTTPException(status_code=500, detail="Inference engine offline: Model artifacts not loaded.")
        
    try:
        # A single vitals snapshot is treated as that parameter's mean/max/min/first/last,
        # matching the representation used at training time
        raw_values = {"Age": payload.Age, "Gender": payload.Gender}
        for obs in payload.Observations:
            raw_values[obs.Parameter] = obs.Value

        feat_dict = {"Age": payload.Age, "Gender": payload.Gender}
        for param, value in raw_values.items():
            if param in ("Age", "Gender"):
                continue
            feat_dict[f"{param}_mean"] = value
            feat_dict[f"{param}_max"] = value
            feat_dict[f"{param}_min"] = value
            feat_dict[f"{param}_first"] = value
            feat_dict[f"{param}_last"] = value
            feat_dict[f"{param}_std"] = 0.0
            feat_dict[f"{param}_count"] = 1.0
            feat_dict[f"{param}_missing"] = 0
            feat_dict[f"{param}_delta"] = 0.0

        # Recompute derived clinical ratios (Shock Index, P/F ratio, etc.) using training-time formulas
        ratio_input_row = pd.DataFrame([feat_dict])
        ratio_input_row = construct_physiological_ratios(ratio_input_row)
        feat_dict.update(ratio_input_row.iloc[0].to_dict())

        # Anything not submitted stays NaN, matching how "never measured" is represented in training
        row_vector = [feat_dict.get(col, np.nan) for col in FEATURE_CHECKLIST]
        X_df = pd.DataFrame([row_vector], columns=FEATURE_CHECKLIST)

        # 3. LightGBM Ensemble Inference Pass
        lgbm_prob = np.mean([model.predict_proba(X_df.values)[:, 1] for model in LGBM_ENSEMBLE], axis=0)[0]

        # 4. PyTorch BiLSTM Inference Pass -- hold each submitted value flat across 48 hours,
        # fall back to clinical defaults for anything missing, then apply training-time normalization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lstm_feature_dim = MODEL_PAYLOAD.get("lstm_feature_count", len(SEQUENCE_FEATURE_NAMES))
        lstm_engine = ClinicalLSTM(n_features=lstm_feature_dim).to(device)
        lstm_engine.load_state_dict(PYTORCH_STATE)
        lstm_engine.eval()

        raw_seq = np.zeros((1, 48, lstm_feature_dim), dtype=np.float32)
        for ch_idx, seq_param in enumerate(SEQUENCE_FEATURE_NAMES):
            value = raw_values.get(seq_param, config.CLINICAL_DEFAULTS.get(seq_param, 0.0))
            raw_seq[0, :, ch_idx] = value
        norm_seq = (raw_seq - SEQ_NORM_MEAN) / SEQ_NORM_STD

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(norm_seq).to(device)
            lstm_logit = float(lstm_engine(seq_tensor).cpu().numpy().reshape(-1)[0])
            lstm_prob = float(1 / (1 + np.exp(-lstm_logit)))

        # 5. Blended Consensus Prediction using the learned stacking weights
        stack_probs = {"lgbm": np.array([lgbm_prob]), "lstm": np.array([lstm_prob])}
        final_prob = float(np.clip(apply_stacking_ensemble(STACKER, STACKER_INPUT_ORDER, stack_probs)[0], 0.0, 1.0))
        
        if final_prob < 0.20:
            status_flag = "LOW RISK"
        elif final_prob < 0.50:
            status_flag = "MODERATE RISK"
        else:
            status_flag = "CRITICAL HIGH RISK"
            
        # Get feature contributions for this prediction.
        shap_raw_vals = SHAP_EXPLAINER.shap_values(X_df)
        
        if isinstance(shap_raw_vals, list):
            shap_values_class1 = shap_raw_vals[1][0] if len(shap_raw_vals) > 1 else shap_raw_vals[0][0]
        elif isinstance(shap_raw_vals, np.ndarray) and shap_raw_vals.ndim == 3:
            shap_values_class1 = shap_raw_vals[0, :, 1]
        else:
            shap_values_class1 = shap_raw_vals[0]
            
        # Match each contribution to its feature name.
        feature_impacts = []
        for idx, impact_val in enumerate(shap_values_class1):
            if idx < len(FEATURE_CHECKLIST):
                raw_name = FEATURE_CHECKLIST[idx]
                clean_name = format_feature_label(raw_name)
                feature_impacts.append((clean_name, float(impact_val)))

        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        escalating = [{"feature": f, "impact": round(v, 3)} for f, v in feature_impacts if v > 0][:3]
        mitigating = [{"feature": f, "impact": round(v, 3)} for f, v in feature_impacts if v < 0][:3]
        
        return {
            "Mortality_Risk_Probability": round(final_prob, 4),
            "Clinical_Status_Flag": status_flag,
            "Review_Suggestions": build_review_suggestions(payload, final_prob),
            "Primary_Risk_Drivers": {
                "escalating": escalating,
                "mitigating": mitigating
            }
        }
        
    except Exception as e:
        api_logger.error(f"Inference execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference execution error: {str(e)}")

@app.post("/rank")
def rank_patients(patients: List[RankedPatientPayload]):
    if not patients:
        raise HTTPException(status_code=400, detail="At least one patient is required.")

    ranked = []
    for item in patients:
        prediction = predict_mortality(item.Patient)
        ranked.append({
            "Patient_ID": item.Patient_ID,
            "Mortality_Risk_Probability": prediction["Mortality_Risk_Probability"],
            "Clinical_Status_Flag": prediction["Clinical_Status_Flag"],
            "Review_Suggestions": prediction["Review_Suggestions"],
        })

    ranked.sort(key=lambda item: item["Mortality_Risk_Probability"], reverse=True)
    for position, item in enumerate(ranked, start=1):
        item["Priority_Rank"] = position
    return {"Patients": ranked}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "icu-mortality-inference-api"}