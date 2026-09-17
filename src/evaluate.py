# src/evaluate.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import config
from src.utils import setup_logger

logger = setup_logger("evaluation_engine")

def calculate_physionet_event1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    PhysioNet Challenge 2012 Event 1: max(min(Sensitivity, PPV)) over all thresholds.
    Returns (score, optimal_threshold).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Handle edge case where thresholds array length is less than precision/recall
    if len(thresholds) == 0:
        return 0.0, 0.5
    scores = np.minimum(precision[:-1], recall[:-1])
    best_idx = np.argmax(scores)
    return float(scores[best_idx]), float(thresholds[best_idx])

def calculate_physionet_event2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    PhysioNet Challenge 2012 Event 2: Modified Hosmer-Lemeshow calibration metric.
    Measures decile-level calibration quality. Lower is better.
    """
    y_prob = np.clip(y_prob, 0.01, 0.99)
    data = pd.DataFrame({'observed': y_true, 'predicted': y_prob})
    data = data.sort_values('predicted').reset_index(drop=True)
    data['decile'] = pd.qcut(data.index, 10, labels=False, duplicates='drop')

    h_stat = 0.0
    decile_means = []
    
    for g in sorted(data['decile'].unique()):
        group = data[data['decile'] == g]
        Og = group['observed'].sum()
        pi_g = group['predicted'].mean()
        Ng = len(group)
        
        decile_means.append(pi_g)
        variance = Ng * pi_g * (1 - pi_g)
        h_stat += (Og - (Ng * pi_g)) ** 2 / (variance + 1e-9)

    if len(decile_means) < 2:
        return float(h_stat)
        
    D = decile_means[-1] - decile_means[0]
    return float(h_stat / (D + 1e-9))

def bootstrap_confidence_interval(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bootstrap: int = 200,  # Balanced for high speed during live training runs
    ci: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """Computes exact 95% Confidence Intervals for AUROC and AUPRC using bootstrap resampling."""
    aurocs, auprcs = [], []
    rng = np.random.RandomState(config.SEED)
    
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0 or (1 - y_true[idx]).sum() == 0:
            continue
        p, r, _ = precision_recall_curve(y_true[idx], y_prob[idx])
        auprcs.append(auc(r, p))
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        
    alpha = (1 - ci) / 2
    return {
        "AUROC_CI": (float(np.quantile(aurocs, alpha)), float(np.quantile(aurocs, 1 - alpha))),
        "AUPRC_CI": (float(np.quantile(auprcs, alpha)), float(np.quantile(auprcs, 1 - alpha)))
    }

def calculate_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes the full clinical performance suite, combining standard and challenge metrics."""
    p, r, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(r, p)
    auroc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    e1_score, e1_thresh = calculate_physionet_event1(y_true, y_prob)
    e2_score = calculate_physionet_event2(y_true, y_prob)
    ci_bounds = bootstrap_confidence_interval(y_true, y_prob)
    
    return {
        "AUROC": float(auroc),
        "AUROC_CI_Lower": ci_bounds["AUROC_CI"][0],
        "AUROC_CI_Upper": ci_bounds["AUROC_CI"][1],
        "AUPRC": float(auprc),
        "AUPRC_CI_Lower": ci_bounds["AUPRC_CI"][0],
        "AUPRC_CI_Upper": ci_bounds["AUPRC_CI"][1],
        "Brier_Loss": float(brier),
        "PhysioNet_Event1": e1_score,
        "PhysioNet_Event1_Threshold": e1_thresh,
        "PhysioNet_Event2": e2_score
    }

def run_stratified_validation(X: pd.DataFrame, y: np.ndarray) -> Tuple[List[CalibratedClassifierCV], np.ndarray, Dict[str, Any]]:
    """
    Executes cross-validation using an ensemble-of-folds strategy with clean prefit calibration.
    Returns: List of trained fold models, out-of-fold risk scores, and summary performance metrics.
    """
    logger.info(f"Starting {config.N_SPLITS}-Fold Stratified Validation Loop.")
    
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    oof_predictions = np.zeros(len(X))
    trained_fold_models: List[CalibratedClassifierCV] = []
    fold_metrics: List[Dict[str, Any]] = []
    
    # FIXED: Keep X_pure as a pandas DataFrame to preserve feature names for LightGBM & SHAP
    feature_cols = [c for c in X.columns if c != 'RecordId']
    X_pure = X[feature_cols] 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pure, y)):
        X_train_fold, y_train_fold = X_pure.iloc[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X_pure.iloc[val_idx], y[val_idx]
        
        # Split training fold into distinct training and calibration subsets
        X_tr, X_calib, y_tr, y_calib = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.15,
            stratify=y_train_fold,
            random_state=config.SEED
        )
        
        # Early stopping uses the calibration split's AUPRC to pick the boosting round
        base_params = {**config.LGBM_PARAMS, 'n_estimators': config.LGBM_MAX_ESTIMATORS}
        base_model = lgb.LGBMClassifier(**base_params)
        base_model.fit(
            X_tr, y_tr,
            eval_set=[(X_calib, y_calib)],
            eval_metric='average_precision',
            callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
        )

        # Fit calibration layer on the un-polluted calibration split using cv='prefit'
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
        
        val_probs = calibrated_model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = val_probs
        trained_fold_models.append(calibrated_model)
        
        f_scores = calculate_clinical_metrics(y_val_fold, val_probs)
        fold_metrics.append(f_scores)
        logger.info(f"Fold {fold+1} -> AUPRC: {f_scores['AUPRC']:.4f} | Event1: {f_scores['PhysioNet_Event1']:.4f}")
        
    # Report performance stability across all folds (Mean ± Std)
    logger.info("--- Cross-Validation Stability Card ---")
    for m in ["AUROC", "AUPRC", "PhysioNet_Event1", "PhysioNet_Event2"]:
        vals = [f[m] for f in fold_metrics]
        logger.info(f" Fold Variance -> Met: {m:<18} | Value: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
    overall_metrics = calculate_clinical_metrics(y, oof_predictions)
    return trained_fold_models, oof_predictions, overall_metrics


def fit_stacking_ensemble(oof_probs: Dict[str, np.ndarray], y_true: np.ndarray) -> Tuple[LogisticRegression, List[str]]:
    """Fits a logistic regression stacker over base-learner OOF predictions. Returns the model
    names in the fixed column order used, so callers apply it consistently at inference time."""
    model_names = sorted(oof_probs.keys())
    stack_X = np.column_stack([oof_probs[name] for name in model_names])
    stacker = LogisticRegression(random_state=config.SEED)
    stacker.fit(stack_X, y_true)
    return stacker, model_names


def apply_stacking_ensemble(stacker: LogisticRegression, model_names: List[str], probs: Dict[str, np.ndarray]) -> np.ndarray:
    """Applies a fitted stacking meta-learner, respecting the exact column order it was fit with."""
    stack_X = np.column_stack([probs[name] for name in model_names])
    return stacker.predict_proba(stack_X)[:, 1]


def evaluate_ensemble(oof_probs: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
    """
    Fits a stacking meta-learner on the base learners' out-of-fold predictions and runs the full
    performance suite on the resulting blended output.
    """
    stacker, model_names = fit_stacking_ensemble(oof_probs, y_true)
    ensemble_probs = apply_stacking_ensemble(stacker, model_names, oof_probs)
    metrics = calculate_clinical_metrics(y_true, ensemble_probs)

    logger.info("==================================================")
    logger.info("   HYBRID ENSEMBLE HYBRID EVALUATION SUCCESSFUL   ")
    logger.info("==================================================")
    for name, coef in zip(model_names, stacker.coef_[0]):
        logger.info(f" Stacker Learned Weight -> {name}: {coef:.4f}")
    logger.info(f" Ensemble Balanced Event1 Score: {metrics['PhysioNet_Event1']:.4f}")
    logger.info(f" Ensemble Calibration Event2 Metric: {metrics['PhysioNet_Event2']:.4f}")

    return {
        "ensemble_probs": ensemble_probs,
        "metrics": metrics,
        "stacker": stacker,
        "stacker_input_order": model_names
    }