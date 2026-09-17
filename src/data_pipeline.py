# src/data_pipeline.py
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import config
from src.utils import setup_logger

logger = setup_logger("data_pipeline")

def process_raw_patient_file(file_path: Path) -> pd.DataFrame:
    """
    Parses a single sparse ASCII patient text file into a clean DataFrame.
    Intercepts explicit missingness markers (-1), empty strings, and text NaN
    signatures to preserve downstream informative missingness signals.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        records = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) == 3:
                param_name = parts[1].strip()
                raw_val = parts[2].strip()
                
                # FIXED: Comprehensive string token matcher preventing runtime float casting failures
                if raw_val in ('-1', '', 'NaN', 'nan', 'None'):
                    val_score = np.nan
                else:
                    try:
                        val_score = float(raw_val)
                    except ValueError:
                        val_score = np.nan
                
                records.append({
                    "Timestamp": parts[0].strip(),
                    "Parameter": param_name,
                    "Value": val_score
                })
                
        return pd.DataFrame(records)
        
    except Exception as e:
        logger.debug(f"Skipping corrupt or unreadable patient ledger {file_path.name}: {str(e)}")
        return pd.DataFrame()

def compile_raw_database(dataset_type: str = "set-a") -> pd.DataFrame:
    """
    Aggregates thousands of patient flat logs into a single long-form
    compressed database dataframe backed by Parquet local cache storage.
    """
    cache_path = config.PROCESSED_DATA_DIR / f"raw_database_cache_{dataset_type.replace('-', '_')}.parquet"
    
    if cache_path.exists():
        logger.info(f"💾 Discovered compiled binary cache snapshot for {dataset_type} at {cache_path}. Loading...")
        return pd.read_parquet(cache_path)
        
    target_dir = config.RAW_DATA_DIR / dataset_type
    if not target_dir.exists():
        logger.error(f"Target data directory context path not found: {target_dir}")
        raise FileNotFoundError(f"Directory {target_dir} is completely missing.")
        
    all_files = list(target_dir.glob("*.txt"))
    logger.info(f"🔍 No cache found. Aggregating {len(all_files)} text logs for {dataset_type}...")
    
    compiled_dfs = []
    for f_path in all_files:
        try:
            record_id = int(f_path.stem)
            patient_df = process_raw_patient_file(f_path)
            if not patient_df.empty:
                patient_df["RecordId"] = record_id
                compiled_dfs.append(patient_df)
        except Exception as e:
            logger.warning(f"Failed parsing file metadata block {f_path.name}: {str(e)}")
            
    if not compiled_dfs:
        raise RuntimeError(f"Data Ingestion cycle collapsed. 0 valid text files were parsed from {target_dir}")
        
    master_long_df = pd.concat(compiled_dfs, ignore_index=True)
    master_long_df.to_parquet(cache_path, compression="snappy")
    logger.info(f"✨ Successfully frozen compressed database snapshot cache at: {cache_path}")
    return master_long_df


def attach_outcomes(features_df: pd.DataFrame, dataset_type: str = "set-a") -> pd.DataFrame:
    """Handshakes advanced extracted features with true mortality target indices."""
    # Dynamic suffix parser supporting set-a, set-b, set-c, etc.
    dataset_code = dataset_type.replace("set-", "").strip().lower()
    outcome_filename = f"Outcomes-{dataset_code}.txt"
    outcome_path = config.RAW_DATA_DIR / outcome_filename
    
    if not outcome_path.exists():
        logger.error(f"Ground truth clinical label target file missing at path: {outcome_path}")
        raise FileNotFoundError(f"Missing registry ledger file: {outcome_filename}")
        
    labels_df = pd.read_csv(outcome_path)
    labels_df.columns = [col.replace('ID', 'Id') for col in labels_df.columns]
    
    target_columns = ["RecordId", "In-hospital_death"]
    labels_subset = labels_df[target_columns].copy()
    
    merged_dataset = pd.merge(features_df, labels_subset, on="RecordId", how="inner")
    logger.info(f"Label alignment complete for {dataset_type}. Matched records cohort count: {merged_dataset.shape[0]}")
    return merged_dataset