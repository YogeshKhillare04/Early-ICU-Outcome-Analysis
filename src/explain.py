# src/explain.py
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

import config
from src.utils import setup_logger

logger = setup_logger("explainability_engine")

def generate_tabular_explanation(
    lgbm_models: List[Any], 
    X_patient: pd.DataFrame, 
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculates consensus TreeSHAP values across all 5 cross-validation LightGBM folds 
    by digging past the CalibratedClassifierCV wrapper layer to extract the raw trees.
    """
    try:
        # Align column ordering strictly with what the models expect
        X_aligned = X_patient[feature_names].copy()
        accumulated_shap = np.zeros(X_aligned.shape)
        
        for model in lgbm_models:
            if hasattr(model, "calibrated_classifiers_"):
                raw_tree_model = model.calibrated_classifiers_[0].estimator
            else:
                raw_tree_model = model
                
            explainer = shap.TreeExplainer(raw_tree_model)
            shap_values = explainer.shap_values(X_aligned)
            
            if isinstance(shap_values, list):
                accumulated_shap += shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                accumulated_shap += shap_values[:, :, 1]
            else:
                accumulated_shap += shap_values
                
        # Compute consensus mean contribution across all fold models
        mean_shap = accumulated_shap / len(lgbm_models)
        
        # Map feature names explicitly to their respective SHAP attribution value
        explanation_map = {}
        for idx, name in enumerate(feature_names):
            explanation_map[str(name)] = float(mean_shap[0][idx])
        
        # Sort values by absolute impact size to present key drivers first
        sorted_explanation = dict(
            sorted(explanation_map.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        
        return sorted_explanation
        
    except Exception as e:
        logger.error(f"Failed to generate tabular SHAP explanations: {str(e)}")
        raise RuntimeError(f"Explainability execution trace fault: {str(e)}")

def isolate_top_risk_drivers(
    shap_dict: Dict[str, float], 
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Filters a raw SHAP mapping dictionary into clean clinical categories 
    showing what factors explicitly increase or decrease patient risk.
    """
    escalating_factors = []
    mitigating_factors = []
    
    for feature, val in shap_dict.items():
        if val > 0.001:
            escalating_factors.append({"feature": feature, "impact": val})
        elif val < -0.001:
            mitigating_factors.append({"feature": feature, "impact": val})
            
    return {
        "escalating": escalating_factors[:top_n],
        "mitigating": mitigating_factors[:top_n]
    }