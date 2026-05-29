import pickle
from pathlib import Path

import numpy as np

# S12 excluded from WESAD (corrupted data)
SUBJECTS = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]

WRIST_FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
LABEL_FS = 700  # labels sampled at 700 Hz in WESAD


def load_subject(data_dir: Path, subject: str) -> dict:
    """
    Load a single WESAD subject pickle (Empatica E4 wrist signals).

    Returns:
        {
          "subject": str,
          "BVP": np.ndarray  shape (N,) at 64 Hz,
          "EDA": np.ndarray  shape (M,) at 4 Hz,
          "labels_700hz": np.ndarray shape (L,) raw WESAD labels,
        }
    """
    pkl_path = data_dir / subject / f"{subject}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]

    return {
        "subject": subject,
        "BVP": wrist["BVP"].flatten().astype(np.float32),
        "EDA": wrist["EDA"].flatten().astype(np.float32),
        "labels_700hz": data["label"].flatten(),
    }
