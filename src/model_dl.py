# src/model_dl.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import logging

import config
from src.utils import setup_logger

logger = setup_logger("pytorch_dl_engine")

class ClinicalLSTM(nn.Module):
    """
    Bidirectional LSTM framework built to capture patient stabilization
    and crashing patterns from irregular time-series ICU sequences.
    """
    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        # Bidirectional layer evaluates trajectories forward and backward
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # Mean+max pooling over all timesteps rather than just the last one -- more robust to
        # sparse, irregularly-sampled ICU readings
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # raw logits; sigmoid is applied by BCEWithLogitsLoss / at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch Size, 48 Hours, Number of Features)
        lstm_out, _ = self.lstm(x)
        mean_pooled = lstm_out.mean(dim=1)
        max_pooled, _ = lstm_out.max(dim=1)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.classifier(pooled).squeeze(1)

def train_lstm(
    sequences: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 30,
    batch_size: int = 64,
    device: str = None
) -> tuple:
    """
    Trains the ClinicalLSTM model across an out-of-fold validation loop.
    Implements early stopping to protect against overfitting.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing PyTorch execution training loops on device target: {device}")

    n_patients, timesteps, n_features = sequences.shape
    oof_probabilities = np.zeros(n_patients)

    # Stratified K-Fold setup to mirror our LightGBM validation splits
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)

    # Calculate pos_weight to penalize missed mortality cases (~14.2% positive balance)
    pos_counts = float(y.sum())
    neg_counts = float(len(y) - pos_counts)
    pos_weight_ratio = torch.tensor([neg_counts / pos_counts], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ratio)

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, y)):
        logger.info(f"--- PyTorch LSTM: Training Fold {fold + 1} / {config.N_SPLITS} ---")

        # Initialize an isolated model instance for this specific cross-validation fold
        model = ClinicalLSTM(n_features=n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max')

        # Allocate Tensor matrices
        X_tr = torch.FloatTensor(sequences[train_idx]).to(device)
        y_tr = torch.FloatTensor(y[train_idx]).to(device)
        X_val = torch.FloatTensor(sequences[val_idx]).to(device)
        y_val_np = y[val_idx]

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

        best_val_auprc = 0.0
        patience_counter = 0
        best_state_weights = model.state_dict().copy()

        for epoch in range(n_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()

                # Gradient clipping protects against exploding weight updates
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation Step
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()

                val_probs = np.nan_to_num(val_probs, nan=0.0, posinf=1.0, neginf=0.0)

            val_auprc = average_precision_score(y_val_np, val_probs)
            scheduler.step(val_auprc)

            # Save the best weights based on validation performance
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 7:  # Trigger early stopping if validation stagnates
                break

        logger.info(f"Fold {fold + 1} Finalized. Peak Validation AUPRC: {best_val_auprc:.4f}")

        # Load the champion weights back into memory to record our out-of-fold predictions
        model.load_state_dict(best_state_weights)
        model.eval()
        with torch.no_grad():
            final_logits = model(X_val)
            probs_vector = torch.sigmoid(final_logits).cpu().numpy()
            oof_probabilities[val_idx] = np.nan_to_num(probs_vector, nan=0.0, posinf=1.0, neginf=0.0)

    # Return a champion model along with the out-of-fold prediction probabilities
    return model, oof_probabilities
