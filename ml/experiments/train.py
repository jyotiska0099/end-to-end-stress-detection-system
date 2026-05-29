"""
LOSO training for StressCNNLSTM with MLflow logging.

Usage:
    python -m ml.experiments.train                      # hold out S17 (default)
    python -m ml.experiments.train --test-subject S5    # hold out a specific subject
    python -m ml.experiments.train --loso               # full LOSO (all 15 subjects)
    python -m ml.experiments.train --data-dir /path/to/wesad
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.data.preprocessor import build_dataset, normalize, window_subject
from ml.data.wesad_loader import SUBJECTS, load_subject
from ml.models.cnn_lstm import StressCNNLSTM

# ── Defaults (override via CLI args) ──────────────────────────────────────────
DATA_DIR = Path("ml/data/wesad")
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────


def windows_to_tensors(windows: list[dict]) -> tuple:
    eda, bvp, labels = build_dataset(windows)
    return (
        torch.tensor(eda),
        torch.tensor(bvp),
        torch.tensor(labels, dtype=torch.float32),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for eda, bvp, labels in loader:
        eda, bvp, labels = eda.to(DEVICE), bvp.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(eda, bvp)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for eda, bvp, labels in loader:
        eda, bvp, labels = eda.to(DEVICE), bvp.to(DEVICE), labels.to(DEVICE)
        logits = model(eda, bvp)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return total_loss / total, correct / total, f1


def run_fold(test_subject: str, data_dir: Path) -> float:
    train_subjects = [s for s in SUBJECTS if s != test_subject]
    print(f"\n{'=' * 60}")
    print(f"  Hold-out: {test_subject}  |  Train: {len(train_subjects)} subjects")
    print(f"{'=' * 60}")

    # ── Load & window ─────────────────────────────────────────────────────────
    print("Loading and windowing subjects...")
    train_windows: list[dict] = []
    for s in train_subjects:
        train_windows.extend(window_subject(load_subject(data_dir, s)))
    test_windows = window_subject(load_subject(data_dir, test_subject))

    tr_eda, tr_bvp, tr_labels = windows_to_tensors(train_windows)
    te_eda, te_bvp, te_labels = windows_to_tensors(test_windows)

    # ── Normalize ─────────────────────────────────────────────────────────────
    tr_eda_n, tr_bvp_n, te_eda_n, te_bvp_n, _, _ = normalize(
        tr_eda.numpy(),
        tr_bvp.numpy(),
        te_eda.numpy(),
        te_bvp.numpy(),
    )
    tr_eda = torch.tensor(tr_eda_n)
    tr_bvp = torch.tensor(tr_bvp_n)
    te_eda = torch.tensor(te_eda_n)
    te_bvp = torch.tensor(te_bvp_n)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(tr_eda, tr_bvp, tr_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(te_eda, te_bvp, te_labels),
        batch_size=BATCH_SIZE,
    )

    # ── Class-weighted loss ───────────────────────────────────────────────────
    n_stress = tr_labels.sum().item()
    n_total = len(tr_labels)
    pos_weight = torch.tensor([(n_total - n_stress) / (n_stress + 1e-8)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model = StressCNNLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"Device: {DEVICE}")
    print(f"Train windows: {len(train_windows)} | Test windows: {len(test_windows)}")
    print(f"Stress ratio (train): {int(n_stress)}/{n_total} ({100 * n_stress / n_total:.1f}%)")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1, best_epoch = 0.0, 0
    best_ckpt = f"best_model_{test_subject}.pt"

    with mlflow.start_run(run_name=f"LOSO_{test_subject}"):
        mlflow.log_params(
            {
                "test_subject": test_subject,
                "n_train_subjects": len(train_subjects),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "device": str(DEVICE),
                "window_sec": 60,
                "stride_sec": 30,
                "n_subwindows": 6,
            }
        )

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            te_loss, te_acc, te_f1 = evaluate(model, test_loader, criterion)
            scheduler.step(te_loss)

            mlflow.log_metrics(
                {
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                    "test_f1": te_f1,
                },
                step=epoch,
            )

            print(
                f"  [{epoch:02d}/{EPOCHS}] "
                f"loss {tr_loss:.4f}→{te_loss:.4f}  "
                f"acc {tr_acc:.3f}→{te_acc:.3f}  "
                f"f1 {te_f1:.3f}  ({time.time() - t0:.1f}s)"
            )

            if te_f1 > best_f1:
                best_f1, best_epoch = te_f1, epoch
                torch.save(model.state_dict(), best_ckpt)

        mlflow.log_artifact(best_ckpt)
        mlflow.log_metrics({"best_f1": best_f1, "best_epoch": best_epoch})
        print(f"\n  ✓ Best F1 = {best_f1:.4f}  (epoch {best_epoch})")

    return best_f1


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    global DATA_DIR, EPOCHS, BATCH_SIZE, LR  # ← moved to top

    parser = argparse.ArgumentParser(description="Train StressCNNLSTM with LOSO CV")
    parser.add_argument(
        "--test-subject", default="S17", choices=SUBJECTS, help="Subject to hold out (default: S17)"
    )
    parser.add_argument("--loso", action="store_true", help="Run full LOSO over all 15 subjects")
    parser.add_argument("--data-dir", default="ml/data/wesad", help="Path to WESAD root directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    mlflow.set_experiment("stress-cnn-lstm-loso")

    if args.loso:
        results = {s: run_fold(s, DATA_DIR) for s in SUBJECTS}
        print("\n── LOSO Summary ──────────────────────────────────────")
        for s, f1 in results.items():
            print(f"  {s}: F1 = {f1:.4f}")
        print(f"  Mean F1 = {np.mean(list(results.values())):.4f}")
    else:
        run_fold(args.test_subject, DATA_DIR)


if __name__ == "__main__":
    main()
