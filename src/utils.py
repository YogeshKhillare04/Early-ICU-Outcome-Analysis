# src/utils.py
import os
import random
import logging
import numpy as np
import joblib
from datetime import datetime
from typing import Any
import config 

def setup_logger(module_name: str) -> logging.Logger:
    """
    Configures a hierarchical module-specific logger pointing to a unified 
    central enterprise logging ledger file to enable sequential run audits.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # FIXED: Indented all initialization logic inside the handler check block
    if not logger.handlers:
        # Fetching the structural blueprint from the main library module safely
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # --- Stream 1: Direct Standard Console Streaming Output ---
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # --- Stream 2: Centralized Consolidated Run File Output ---
        log_dir = config.ROOT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        central_log_file = log_dir / "Pipeline_execution.log"
        file_handler = logging.FileHandler(central_log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Initialize module-level diagnostic auditor safely
utils_logger = setup_logger("utils_infra")

def enforce_reproducibility(seed: int | None = None) -> None:
    """
    Enforces total system determinism by freezing pseudo-random engines across
    core Python, NumPy, and PyTorch mathematical computing execution frames.
    """
    target_seed = seed if seed is not None else config.SEED
    random.seed(target_seed)
    os.environ['PYTHONHASHSEED'] = str(target_seed)
    np.random.seed(target_seed)

    # also seed torch, for LSTM weight init and DataLoader shuffling
    try:
        import torch
        torch.manual_seed(target_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(target_seed)
    except ImportError:
        pass

    utils_logger.info(f"Global pseudorandom states locked securely using seed footprint: {target_seed}")
    utils_logger.warning("Reminder: Estimator-level routines require internal 'random_state' parameter assignment.")

def save_artifact(artifact: Any, filename: str) -> None:
    """
    Serializes an active state Python or Scikit-Learn machine learning artifact 
    safely down to structural memory storage targets on disk.
    """
    try:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        target_path = config.MODEL_DIR / filename

        joblib.dump(artifact, target_path, compress=3)
        utils_logger.info(f"Successfully serialized production asset artifact to disk destination: {target_path}")
    except Exception as e:
        # FIXED: Corrected error messaging to reflect a serialization fault
        utils_logger.error(f"Serialization mapping sequence failed for target resource {filename}: {str(e)}")  
        raise RuntimeError(f"Corrupt binary framework allocation block detected: {str(e)}") 

def load_artifact(filename: str) -> Any:
    """
    De-serializes a frozen binary machine learning weight parameter array block 
    from persistent file systems back up into system runtime RAM memory frames.
    """
    target_path = config.MODEL_DIR / filename
    if not target_path.exists():
        utils_logger.error(f"Requested production artifact file is missing from target path structure: {target_path}")
        raise FileNotFoundError(f"Requested clinical operational state template {filename} cannot be located.")
        
    try:
        loaded_asset = joblib.load(target_path)
        utils_logger.info(f"Successfully loaded and verified structure for asset file: {filename}")
        return loaded_asset
    except Exception as e:
        utils_logger.error(f"De-serialization mapping sequence failed for target resource {filename}: {str(e)}")
        raise RuntimeError(f"Corrupt binary framework allocation block detected: {str(e)}")


# ==============================================================================
# CUSTOM CLINICAL SYSTEM EXCEPTIONS
# ==============================================================================

class ICUSystemException(Exception):
    """Base exception class for errors in the Early ICU Outcome Analysis pipeline."""
    pass

class EmptyPatientRecordError(ICUSystemException):
    """Raised when a patient text file contains no valid observations inside the 48h window."""
    def __init__(self, record_id: int, message: str = "No observations found within 48-hour window"):
        self.record_id = record_id
        self.message = f"RecordId {record_id}: {message}"
        super().__init__(self.message)

class MissingDemographicsError(ICUSystemException):
    """Raised when critical baseline profiles (Age, Gender) are completely absent."""
    def __init__(self, record_id: int, missing_metric: str):
        self.record_id = record_id
        self.message = f"RecordId {record_id}: Critical demographic baseline metric '{missing_metric}' is missing."
        super().__init__(self.message)