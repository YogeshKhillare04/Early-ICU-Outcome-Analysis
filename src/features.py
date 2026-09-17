# src/features.py
import pandas as pd
import numpy as np
from typing import Dict, Any

import config
from src.utils import setup_logger

logger = setup_logger("feature_engineering")

CLINICAL_VOCABULARY = {
    "Age": "Patient Age",
    "Gender": "Biological Gender Assignment",
    "Height": "Patient Height",
    "HR": "Heart Rate Tracking",
    "GCS": "Glasgow Coma Scale",
    "SysABP": "Systolic Blood Pressure (Invasive)",
    "NISysABP": "Systolic Blood Pressure (Non-Invasive)",
    "DiasABP": "Diastolic Blood Pressure",
    "MAP": "Mean Arterial Pressure",
    "Temp": "Core Body Temperature",
    "RespRate": "Respiration Rate",
    "BUN": "Blood Urea Nitrogen Labs",
    "Creatinine": "Serum Creatinine Levels",
    "Platelets": "Platelet Concentration",
    "WBC": "White Blood Cell Count",
    "Glucose": "Serum Glucose Baseline",
    "FiO2": "Fractional Inspired Oxygen Vent",
    "PaO2": "Partial Pressure of Oxygen",
    "PaCO2": "Partial Pressure of Carbon Dioxide",
    "HCO3": "Serum Bicarbonate",
    "Lactate": "Serum Lactate",
    "Urine": "Urine Output"
}

AGG_DESCRIPTORS = {
    "mean": "Average", "max": "Peak Max", "min": "Minimum Floor",
    "std": "Variability", "count": "Observation Count"
}

# Vitals used for the 0-6h vs last-6h trajectory deltas and instability score
TRAJECTORY_VITALS = ['HR', 'SysABP', 'MAP', 'RespRate', 'Temp', 'GCS']


def _standardize_time_columns(ts_data: pd.DataFrame) -> pd.DataFrame:
    """Normalizes raw Timestamp/Time text fields into a numeric 'minutes' column."""
    if 'Timestamp' in ts_data.columns and 'Time' not in ts_data.columns:
        ts_data = ts_data.rename(columns={'Timestamp': 'Time'})

    if 'Parameter' in ts_data.columns:
        ts_data['Parameter'] = ts_data['Parameter'].astype(str).str.strip()

    if 'minutes' not in ts_data.columns:
        def standardized_time_to_minutes(val):
            if pd.isna(val): return 0
            if isinstance(val, (int, float)): return int(val)
            val_str = str(val).strip()
            if ':' in val_str:
                try:
                    parts = val_str.split(':')
                    return int(parts[0]) * 60 + int(parts[1])
                except Exception: return 0
            else:
                try: return int(float(val_str))
                except Exception: return 0
        ts_data['minutes'] = ts_data['Time'].apply(standardized_time_to_minutes)

    return ts_data


def _apply_safety_bounds(vitals_data: pd.DataFrame) -> pd.DataFrame:
    """Nulls out physiologically impossible readings (sensor errors / typos) per config bounds."""
    vitals_data = vitals_data.copy()
    for param, (lo, hi) in config.CLINICAL_SAFETY_BOUNDS.items():
        mask = (vitals_data['Parameter'] == param) & (
            (vitals_data['Value'] < lo) | (vitals_data['Value'] > hi)
        )
        vitals_data.loc[mask, 'Value'] = np.nan
    return vitals_data.dropna(subset=['Value'])


def _build_trajectory_deltas(vitals_data: pd.DataFrame, active_params: list) -> pd.DataFrame:
    """Vectorized early-window (0-6h) vs late-window (last 6h) mean deltas per patient/parameter."""
    first_6h = vitals_data[vitals_data['minutes'] <= 360]

    max_times = vitals_data.groupby('RecordId')['minutes'].max().rename('max_time')
    vitals_with_max = vitals_data.join(max_times, on='RecordId')
    last_6h = vitals_with_max[vitals_with_max['minutes'] >= (vitals_with_max['max_time'] - 360)]

    start_means = first_6h.groupby(['RecordId', 'Parameter'])['Value'].mean().unstack()
    end_means = last_6h.groupby(['RecordId', 'Parameter'])['Value'].mean().unstack()

    start_means = start_means.reindex(columns=active_params)
    end_means = end_means.reindex(columns=active_params)

    delta_df = (end_means - start_means)
    delta_df.columns = [f"{col}_delta" for col in delta_df.columns]
    return delta_df


def _build_sequence_cube(vitals_data: pd.DataFrame, patient_index: pd.Index) -> np.ndarray:
    """Builds a (n_patients, 48, n_sequence_features) tensor for the LSTM, forward/back-filled
    per patient so it never sees NaNs. Left unnormalized; normalization happens downstream."""
    n_patients = len(patient_index)
    seq_features = config.SEQUENCE_FEATURES
    cube = np.full((n_patients, 48, len(seq_features)), np.nan, dtype=np.float64)

    vd = vitals_data[vitals_data['Parameter'].isin(seq_features)].copy()
    vd['hour'] = (vd['minutes'] // 60).clip(0, 47).astype(int)

    for j, param in enumerate(seq_features):
        p_rows = vd[vd['Parameter'] == param]
        if p_rows.empty:
            continue
        hourly = p_rows.groupby(['RecordId', 'hour'])['Value'].mean().unstack()
        hourly = hourly.reindex(index=patient_index, columns=range(48))
        cube[:, :, j] = hourly.values

    for j, param in enumerate(seq_features):
        channel = pd.DataFrame(cube[:, :, j])
        channel = channel.ffill(axis=1)
        default_val = config.CLINICAL_DEFAULTS.get(param, np.nanmean(cube[:, :, j]) if not np.all(np.isnan(cube[:, :, j])) else 0.0)
        channel = channel.bfill(axis=1).fillna(default_val)
        cube[:, :, j] = channel.values

    return cube


def construct_physiological_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Builds non-linear medical risk ratios from the raw aggregated mean columns."""
    epsilon = 1e-9
    logger.info("Calculating advanced clinical metric indexes...")

    def col(name, default):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    hr = col('HR_mean', config.CLINICAL_DEFAULTS.get('HR', 80.0)).fillna(config.CLINICAL_DEFAULTS.get('HR', 80.0))
    sys_bp = col('SysABP_mean', np.nan)
    if 'NISysABP_mean' in df.columns:
        sys_bp = sys_bp.fillna(df['NISysABP_mean'])
    sys_bp = sys_bp.fillna(120.0)

    pa_o2 = col('PaO2_mean', config.CLINICAL_DEFAULTS.get('PaO2', 100.0)).fillna(config.CLINICAL_DEFAULTS.get('PaO2', 100.0))
    fi_o2 = col('FiO2_mean', config.CLINICAL_DEFAULTS.get('FiO2', 0.21)).fillna(config.CLINICAL_DEFAULTS.get('FiO2', 0.21))
    bun = col('BUN_mean', 20.0).fillna(20.0)
    creat = col('Creatinine_mean', 1.0).fillna(1.0)
    gcs = col('GCS_mean', config.CLINICAL_DEFAULTS.get('GCS', 15.0)).fillna(config.CLINICAL_DEFAULTS.get('GCS', 15.0))
    gcs_delta = col('GCS_delta', 0.0).fillna(0.0)

    delta_cols = [f"{p}_delta" for p in TRAJECTORY_VITALS if f"{p}_delta" in df.columns]
    total_instability = df[delta_cols].abs().fillna(0.0).sum(axis=1) if delta_cols else pd.Series(0.0, index=df.index)

    ratio_cols = pd.DataFrame({
        'Shock_Index': hr / (sys_bp + epsilon),
        'RPP': hr * sys_bp,
        'PF_Ratio': pa_o2 / (fi_o2 + epsilon),
        'BUN_Creat_Ratio': bun / (creat + epsilon),
        'GCS_Trend_Severity': gcs * gcs_delta,
        'Total_Instability_Score': total_instability,
    }, index=df.index)

    return pd.concat([df, ratio_cols], axis=1)


def extract_advanced_clinical_features(raw_long_df: pd.DataFrame, return_sequences: bool = True) -> Dict[str, Any]:
    """Transforms irregular time-series EHR records into a tabular feature matrix plus an
    optional 3D sequence tensor for the LSTM. Stat columns stay NaN (not zero-filled) where a
    parameter was never measured -- LightGBM splits on missingness natively."""
    logger.info(f"Initializing multi-phase extraction pipeline on matrix shape: {raw_long_df.shape}")

    ts_data = _standardize_time_columns(raw_long_df.copy())
    ts_data = ts_data[ts_data['minutes'] <= config.OBSERVATION_WINDOW_MINUTES].copy()
    if ts_data.empty:
        raise ValueError("No observations remain inside the configured observation window.")

    static_params = ['Age', 'Gender', 'Height', 'ICUType']
    static_data = ts_data[ts_data['Parameter'].isin(static_params)].copy()
    vitals_data_raw = ts_data[~ts_data['Parameter'].isin(static_params)].copy()

    logger.info("Applying clinical safety-bound outlier clipping...")
    vitals_data = _apply_safety_bounds(vitals_data_raw)

    active_params = sorted(vitals_data['Parameter'].unique().tolist())

    logger.info("Aggregating baseline chronological matrices...")
    # Sort by time so 'first'/'last' below are chronologically correct, not just row order
    vitals_data_sorted = vitals_data.sort_values(['RecordId', 'minutes'])
    vitals_summary = vitals_data_sorted.groupby(['RecordId', 'Parameter'])['Value'].agg(
        ['mean', 'max', 'min', 'std', 'count', 'first', 'last']
    ).unstack()
    vitals_summary.columns = [f"{param}_{stat}" for stat, param in vitals_summary.columns]

    logger.info("Executing vectorized timeline trajectory comparisons...")
    delta_df = _build_trajectory_deltas(vitals_data, active_params)

    logger.info("Assembling structured missingness indicator matrix layers...")
    recorded_counts = vitals_data.groupby(['RecordId', 'Parameter']).size().unstack(fill_value=0)
    recorded_counts = recorded_counts.reindex(columns=active_params, fill_value=0)
    missing_df = (recorded_counts == 0).astype(int)
    missing_df.columns = [f"{col}_missing" for col in missing_df.columns]

    static_pivot = static_data.groupby(['RecordId', 'Parameter'])['Value'].mean().unstack()

    all_ids = ts_data['RecordId'].unique()
    tabular_matrix = pd.DataFrame(index=pd.Index(all_ids, name='RecordId'))
    tabular_matrix = tabular_matrix.join(static_pivot).join(vitals_summary).join(delta_df).join(missing_df)

    if 'Age' not in tabular_matrix.columns: tabular_matrix['Age'] = np.nan
    if 'Gender' not in tabular_matrix.columns: tabular_matrix['Gender'] = np.nan
    tabular_matrix['Age'] = tabular_matrix['Age'].fillna(65.0)
    tabular_matrix['Gender'] = tabular_matrix['Gender'].fillna(0.0)

    std_cols = [c for c in tabular_matrix.columns if c.endswith('_std')]
    tabular_matrix[std_cols] = tabular_matrix[std_cols].fillna(0.0)
    count_cols = [c for c in tabular_matrix.columns if c.endswith('_count')]
    tabular_matrix[count_cols] = tabular_matrix[count_cols].fillna(0.0)
    delta_cols_all = [c for c in tabular_matrix.columns if c.endswith('_delta')]
    tabular_matrix[delta_cols_all] = tabular_matrix[delta_cols_all].fillna(0.0)

    tabular_matrix = construct_physiological_ratios(tabular_matrix)
    tabular_matrix = tabular_matrix.drop(columns=config.HIGH_MISSING_DROP_COLS, errors='ignore')

    sequences_cube = None
    if return_sequences:
        logger.info("return_sequences parameter verified. Compiling 3D temporal arrays...")
        sequences_cube = _build_sequence_cube(vitals_data, tabular_matrix.index)

    logger.info("Advanced clinical feature construction sequence finalized successfully.")
    return {
        "tabular": tabular_matrix.reset_index(),
        "sequences": sequences_cube if return_sequences else np.array([])
    }
