# config.py
"""
Project ICU: Centralized System Configuration Framework
Acts as the single source of truth for directory mapping, clinical parameters,
missingness filters, and model hyperparameters.
"""

from pathlib import Path

# ==============================================================================
# 1. PHYSICAL DIRECTORY PATH CONTEXTS
# ==============================================================================
# Keep generated data, models, and logs inside this repository.
ROOT_DIR = Path(__file__).resolve().parent

# Input layers for unzipped files and outcome indices
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

# Output layer for processed data tables, vectors, or arrays
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Persistent storage engine for frozen machine learning binary weights (.joblib files)
MODEL_DIR = ROOT_DIR / "models"


# ==============================================================================
# 2. CLINICAL COHORT WINDOW SETTINGS
# ==============================================================================
# The prospective observation window cutoff (Exactly 48 Hours in minutes)
# Any medical vitals recorded after this marker are dropped to prevent target leakage
OBSERVATION_WINDOW_MINUTES = 2880


# ==============================================================================
# 3. VECTORIZED PHYSIOLOGICAL SAFETY BOUNDS
# ==============================================================================
# Clinical validation limits used by src/features.py to clip extreme telemetry outliers.
# Values outside these ranges represent sensor disconnections or typing anomalies.
CLINICAL_SAFETY_BOUNDS = {
    'Age': [15.0, 110.0],
    'Height': [100.0, 250.0],
    'Weight': [30.0, 300.0],
    'Gender': [0.0, 1.0],
    'HR': [30.0, 220.0],
    'Temp': [25.0, 45.0],
    'GCS': [3.0, 15.0],
    'BUN': [1.0, 250.0],
    'Creatinine': [0.1, 20.0],
    'Urine': [0.0, 10000.0],
    'HCT': [5.0, 70.0],
    'HCO3': [2.0, 60.0],
    'Lactate': [0.1, 30.0],
    'K': [1.5, 10.0],
    'Na': [100.0, 180.0],
    'Mg': [0.5, 10.0],
    'Platelets': [2.0, 1200.0],
    'PaO2': [20.0, 700.0],
    'PaCO2': [10.0, 150.0],
    'SysABP': [40.0, 280.0],
    'DiasABP': [20.0, 160.0],
    'MAP': [30.0, 200.0],
    'RespRate': [4.0, 60.0],
    'WBC': [0.1, 100.0],
    'ALP': [1.0, 2500.0],
    'ALT': [1.0, 2500.0],
    'AST': [1.0, 2500.0],
    'Bilirubin': [0.1, 50.0],
    'SaO2': [50.0, 100.0],
    'Albumin': [1.0, 6.0],
    'FiO2': [0.21, 1.0],
    'Glucose': [10.0, 1000.0],
    'SerumGlc': [10.0, 1000.0],
    'Cholesterol': [50.0, 60.0],
    'TroponinI': [0.01, 50.0],
    'TroponinT': [0.01, 50.0]
}


# ==============================================================================
# 4. SURGICAL IMPUTATION STANDARDS & FALLBACKS
# ==============================================================================
# Healthy physiological baseline values used for conditional filling when a metric 
# is missing and cannot be computed from historical trends.
CLINICAL_DEFAULTS = {
    'HR': 80.0,
    'Lactate': 1.0,
    'Bilirubin': 0.8,
    'Albumin': 4.0,
    'TroponinT': 0.01,
    'SaO2': 98.0,
    'GCS': 15.0,
    'FiO2': 0.21,
    'RespRate': 16.0,
    'MechVent': 0.0,
    'ALP': 40.0,
    'ALT': 10.0,
    'AST': 10.0,
    'PaO2': 100.0,
    'Glucose': 90.0,
    'SerumGlc': 90.0
}


# ==============================================================================
# 5. DATA CLEANING & DIMENSIONALITY FILTERS
# ==============================================================================
# High-missingness variables dropped entirely from summary matrices.
# These parameters have >90% missing rates, offering no statistical value.
HIGH_MISSING_DROP_COLS = [
    'Cholesterol_min', 'Cholesterol_max', 'Cholesterol_mean', 'Cholesterol_delta',
    'TroponinI_min', 'TroponinI_max', 'TroponinI_mean', 'TroponinI_delta',
    'TroponinT_min', 'TroponinT_max', 'TroponinT_mean', 'TroponinT_delta'
]


# ==============================================================================
# 6. DEEP LEARNING COMPONENT VARIABLES
# ==============================================================================
# Core high-frequency physiological parameters tracked step-by-step
# to construct the 3D sequence tensor for LSTM/GRU model variations.
SEQUENCE_FEATURES = [
    'HR', 'SysABP', 'DiasABP', 'MAP', 'GCS', 'RespRate', 
    'Temp', 'Urine', 'PaO2', 'FiO2', 'Lactate', 'Glucose'
]


# ==============================================================================
# 7. DETERMINISTIC REPRODUCIBILITY & VALIDATION PARAMETERS
# ==============================================================================
# Fixed structural random state to anchor pseudo-random generation engines
SEED = 42

# Number of training validation slices to run during cross-validation
N_SPLITS = 5

# Proportion of the full cohort held out as a final untouched prospective test set
TEST_SIZE_PROPORTION = 0.15

# Tuned via a randomized search in src/tune_lgbm.py
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'average_precision',  # Native LightGBM metric name for Area Under the Precision-Recall Curve
    'boosting_type': 'gbdt',
    'scale_pos_weight': 7.0,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 40,
    'feature_fraction': 0.5,
    'bagging_fraction': 1.0,
    'bagging_freq': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': SEED       # Passes seed parameters securely into underlying parallel worker states
}

# Early stopping picks the actual best round, so this just needs to be a safe upper bound
LGBM_MAX_ESTIMATORS = 3000
LGBM_EARLY_STOPPING_ROUNDS = 100