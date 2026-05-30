"""
Model loader and inference logic.

Loads:
  - StressCNNLSTM state dict (.pt)
  - EDA + BVP StandardScalers (.pkl)

from a local MLflow run artifact directory.
"""

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import torch

from app.config import settings
from ml.data.preprocessor import BVP_SUB, EDA_SUB, N_SUBWINDOWS
from ml.models.cnn_lstm import StressCNNLSTM

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_model: StressCNNLSTM | None = None
_eda_scaler = None
_bvp_scaler = None
_run_id: str = ""
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _artifact_path(run_id: str, artifact_name: str) -> Path:
    """Resolve local MLflow artifact path directly from disk."""
    tracking_uri = settings.mlflow_tracking_uri
    # Walk all experiments to find the run directory
    tracking_path = Path(tracking_uri)
    for exp_dir in tracking_path.iterdir():
        if not exp_dir.is_dir():
            continue
        run_dir = exp_dir / run_id / "artifacts" / artifact_name
        if run_dir.exists():
            return run_dir
    raise FileNotFoundError(
        f"Artifact '{artifact_name}' not found for run '{run_id}' "
        f"under tracking URI '{tracking_uri}'"
    )


def load_model() -> None:
    """Load model weights and scalers from MLflow artifact store."""
    global _model, _eda_scaler, _bvp_scaler, _run_id

    run_id = settings.mlflow_run_id
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID not set. Run training first, then set the env var.")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("Loading model from run %s ...", run_id)

    # ── Weights ────────────────────────────────────────────────────────────────
    model_path = _artifact_path(run_id, settings.model_artifact_name)
    model = StressCNNLSTM()
    model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
    model.to(_device)
    model.eval()

    # ── Scalers ────────────────────────────────────────────────────────────────
    eda_path = _artifact_path(run_id, settings.eda_scaler_artifact)
    bvp_path = _artifact_path(run_id, settings.bvp_scaler_artifact)
    with open(eda_path, "rb") as f:
        eda_scaler = pickle.load(f)
    with open(bvp_path, "rb") as f:
        bvp_scaler = pickle.load(f)

    _model = model
    _eda_scaler = eda_scaler
    _bvp_scaler = bvp_scaler
    _run_id = run_id
    logger.info("Model loaded on %s", _device)


def is_loaded() -> bool:
    return _model is not None


def predict(eda_raw: list[float], bvp_raw: list[float]) -> tuple[float, int]:
    """
    Run inference on a single 60s window.

    Args:
        eda_raw: 240 EDA samples
        bvp_raw: 3840 BVP samples

    Returns:
        (stress_probability, label)
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # ── Reshape to sub-windows: (1, 6, sub_len) ───────────────────────────────
    eda_np = np.array(eda_raw, dtype=np.float32).reshape(1, N_SUBWINDOWS, EDA_SUB)
    bvp_np = np.array(bvp_raw, dtype=np.float32).reshape(1, N_SUBWINDOWS, BVP_SUB)

    # ── Normalize using training scalers ──────────────────────────────────────
    eda_np = _eda_scaler.transform(eda_np.reshape(-1, EDA_SUB)).reshape(1, N_SUBWINDOWS, EDA_SUB)
    bvp_np = _bvp_scaler.transform(bvp_np.reshape(-1, BVP_SUB)).reshape(1, N_SUBWINDOWS, BVP_SUB)

    eda_t = torch.tensor(eda_np).to(_device)
    bvp_t = torch.tensor(bvp_np).to(_device)

    with torch.no_grad():
        logit = _model(eda_t, bvp_t)
        prob = torch.sigmoid(logit).item()

    return prob, int(prob > 0.5)
