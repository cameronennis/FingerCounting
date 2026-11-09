"""Train the finger-count classifier from data/labels.csv.

  1. Load every saved sample and normalize its joint positions (see common.py).
  2. Hold back 20% of each label as a test set the model never trains on.
  3. Train the small network (FingerMLP), then report accuracy on the test set.
  4. Save the trained model to models/finger_mlp.pt for live.py.
"""

import csv

import numpy as np
import torch
from torch import nn

from common import (
    CLASSIFIER_PATH, LABELS_CSV, NUM_CLASSES, NUM_LANDMARKS, FingerMLP, normalize_landmarks,
)

TEST_FRACTION = 0.2
EPOCHS = 2000         # one epoch = one pass over all training samples
LEARNING_RATE = 1e-3
SEED = 0              # fixed seed = the same split and result every run


def load_dataset():
    """Read labels.csv -> X (N x 63 normalized features), y (N labels)."""
    if not LABELS_CSV.exists():
        raise SystemExit(f"No data yet: {LABELS_CSV} not found. Run collect.py first.")
    X, y = [], []
    coord_cols = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            raw = np.array([float(row[c]) for c in coord_cols], dtype=np.float32).reshape(NUM_LANDMARKS, 3)
            X.append(normalize_landmarks(raw, row["handedness"], int(row["img_w"]), int(row["img_h"])))
            y.append(int(row["label"]))
    if not y:
        raise SystemExit(f"{LABELS_CSV} has no samples yet. Run collect.py and press 0-5 to save some.")
    return np.stack(X), np.array(y)


def split_per_label(y, rng):
    """Pick test samples separately within each label, so every label is in both sets."""
    train_idx, test_idx = [], []
    for label in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_test = int(round(len(idx) * TEST_FRACTION))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    X, y = load_dataset()
    counts = np.bincount(y, minlength=NUM_CLASSES)
    print(f"Loaded {len(y)} samples. Per label: {dict(enumerate(counts.tolist()))}")
    missing = [i for i, c in enumerate(counts) if c < 5]
    if missing:
        print(f"WARNING: labels {missing} have fewer than 5 samples; results for them will be poor.")

    train_idx, test_idx = split_per_label(y, rng)
    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_test, y_test = torch.tensor(X[test_idx]), torch.tensor(y[test_idx])

    model = FingerMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # The dataset is tiny, so each step uses all training samples at once.
    for epoch in range(1, EPOCHS + 1):
        model.train()                       # dropout on
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()                     # work out how to nudge each weight
        optimizer.step()                    # nudge them
        if epoch % 250 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  "
                  f"train acc {accuracy(model, X_train, y_train):.1%}")

    # Evaluate on the held-back test set
    if len(test_idx):
        test_acc = accuracy(model, X_test, y_test)
        print(f"\nTest accuracy: {test_acc:.1%}  ({len(test_idx)} samples never seen in training)")
        print_confusion(model, X_test, y_test)

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_PATH)
    print(f"\nSaved model to {CLASSIFIER_PATH}")


@torch.no_grad()
def predict(model, X):
    model.eval()                            # dropout off
    return model(X).argmax(dim=1)


def accuracy(model, X, y) -> float:
    return (predict(model, X) == y).float().mean().item()


def print_confusion(model, X, y):
    """Rows = true label, columns = predicted label. Off-diagonal numbers are mistakes."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(y.tolist(), predict(model, X).tolist()):
        matrix[true, pred] += 1
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("      " + " ".join(f"{i:>4}" for i in range(NUM_CLASSES)))
    for i, row in enumerate(matrix):
        print(f"  {i:>2}  " + " ".join(f"{v:>4}" for v in row))


if __name__ == "__main__":
    main()
