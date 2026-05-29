import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Sampling rates ────────────────────────────────────────────────────────────
EDA_FS = 4
BVP_FS = 64
LABEL_FS = 700

# ── Window config ─────────────────────────────────────────────────────────────
WINDOW_SEC = 60
SUB_WINDOW_SEC = 10
STRIDE_SEC = 30  # 50% overlap → more training samples

N_SUBWINDOWS = WINDOW_SEC // SUB_WINDOW_SEC  # 6
EDA_WIN = WINDOW_SEC * EDA_FS  # 240
BVP_WIN = WINDOW_SEC * BVP_FS  # 3840
EDA_SUB = SUB_WINDOW_SEC * EDA_FS  # 40
BVP_SUB = SUB_WINDOW_SEC * BVP_FS  # 640
EDA_STRIDE = STRIDE_SEC * EDA_FS  # 120
BVP_STRIDE = STRIDE_SEC * BVP_FS  # 1920

# ── WESAD raw label mapping ───────────────────────────────────────────────────
WESAD_TRANSITION = 0
WESAD_STRESS = 2
WESAD_NO_STRESS = {1, 3, 4}  # baseline, amusement, meditation

# ── Binary output labels ──────────────────────────────────────────────────────
LABEL_NO_STRESS = 0
LABEL_STRESS = 1


def _align_labels_to_signal(labels_700hz: np.ndarray, signal_fs: int, n_samples: int) -> np.ndarray:
    """Downsample 700 Hz labels to match a wrist signal at signal_fs."""
    ratio = LABEL_FS / signal_fs
    idx = np.clip((np.arange(n_samples) * ratio).astype(int), 0, len(labels_700hz) - 1)
    return labels_700hz[idx]


def window_subject(subject_data: dict) -> list[dict]:
    """
    Slice EDA & BVP into overlapping 60s windows with 30s stride.
    Drops windows containing any transition label (WESAD label 0).
    Assigns binary label by majority vote.

    Returns list of {"EDA": (240,), "BVP": (3840,), "label": int}
    """
    eda = subject_data["EDA"]
    bvp = subject_data["BVP"]
    labels_raw = subject_data["labels_700hz"]

    eda_labels = _align_labels_to_signal(labels_raw, EDA_FS, len(eda))

    windows = []
    n_windows = (len(eda) - EDA_WIN) // EDA_STRIDE + 1

    for i in range(n_windows):
        e0, e1 = i * EDA_STRIDE, i * EDA_STRIDE + EDA_WIN
        b0, b1 = i * BVP_STRIDE, i * BVP_STRIDE + BVP_WIN

        if e1 > len(eda) or b1 > len(bvp):
            break

        win_labels = eda_labels[e0:e1]

        # Drop windows containing transitions
        if np.any(win_labels == WESAD_TRANSITION):
            continue

        # Majority vote
        values, counts = np.unique(win_labels, return_counts=True)
        majority = int(values[np.argmax(counts)])

        if majority == WESAD_STRESS:
            label = LABEL_STRESS
        elif majority in WESAD_NO_STRESS:
            label = LABEL_NO_STRESS
        else:
            continue

        windows.append(
            {
                "EDA": eda[e0:e1].copy(),
                "BVP": bvp[b0:b1].copy(),
                "label": label,
            }
        )

    return windows


def build_dataset(windows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack windows into arrays shaped for the dual-branch CNN-LSTM.

    Returns:
        eda    (N, N_SUBWINDOWS, EDA_SUB)   →  (N, 6, 40)
        bvp    (N, N_SUBWINDOWS, BVP_SUB)   →  (N, 6, 640)
        labels (N,)
    """
    eda_list, bvp_list, label_list = [], [], []
    for w in windows:
        eda_list.append(w["EDA"].reshape(N_SUBWINDOWS, EDA_SUB))
        bvp_list.append(w["BVP"].reshape(N_SUBWINDOWS, BVP_SUB))
        label_list.append(w["label"])

    return (
        np.array(eda_list, dtype=np.float32),
        np.array(bvp_list, dtype=np.float32),
        np.array(label_list, dtype=np.int64),
    )


def normalize(
    train_eda: np.ndarray,
    train_bvp: np.ndarray,
    test_eda: np.ndarray,
    test_bvp: np.ndarray,
) -> tuple:
    """
    Fit StandardScaler on training windows, apply to test.
    Operates on the sub-window level (flattening N×T → N*T rows).

    Returns:
        train_eda_norm, train_bvp_norm, test_eda_norm, test_bvp_norm,
        eda_scaler, bvp_scaler
    """
    N_tr, T, eda_sub = train_eda.shape
    N_te = test_eda.shape[0]
    bvp_sub = train_bvp.shape[2]

    eda_scaler = StandardScaler()
    train_eda_n = eda_scaler.fit_transform(train_eda.reshape(-1, eda_sub)).reshape(N_tr, T, eda_sub)
    test_eda_n = eda_scaler.transform(test_eda.reshape(-1, eda_sub)).reshape(N_te, T, eda_sub)

    bvp_scaler = StandardScaler()
    train_bvp_n = bvp_scaler.fit_transform(train_bvp.reshape(-1, bvp_sub)).reshape(N_tr, T, bvp_sub)
    test_bvp_n = bvp_scaler.transform(test_bvp.reshape(-1, bvp_sub)).reshape(N_te, T, bvp_sub)

    return train_eda_n, train_bvp_n, test_eda_n, test_bvp_n, eda_scaler, bvp_scaler
