"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    confidences = probs.max(axis=1)
    predicted   = probs.argmax(axis=1)
    correct     = (predicted == y_true)

    bucket_accuracies = np.zeros(n_bins)
    bucket_counts     = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        else:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])

        bucket_counts[i] = mask.sum()
        if bucket_counts[i] > 0:
            bucket_accuracies[i] = correct[mask].mean()
        else:
            bucket_accuracies[i] = 0.0

    return centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.
    """
    edges       = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = probs.max(axis=1)
    predicted   = probs.argmax(axis=1)
    correct     = (predicted == y_true)
    N           = len(y_true)

    ece = 0.0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        else:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])

        count = mask.sum()
        if count == 0:
            continue

        bucket_conf = confidences[mask].mean()
        bucket_acc  = correct[mask].mean()
        ece += (count / N) * abs(bucket_acc - bucket_conf)

    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os

    probs  = np.load("manual_probs.npy")
    y_true = np.load("manual_ytrue.npy")

    centers, accs, counts = reliability_diagram(probs, y_true, n_bins=10)
    ece = expected_calibration_error(probs, y_true, n_bins=10)

    print(f"ECE = {ece:.4f}\n")
    print(f"{'Bucket':>8}  {'Accuracy':>9}  {'Count':>6}")
    for c, a, n in zip(centers, accs, counts):
        print(f"{c:8.2f}  {a:9.4f}  {n:6d}")

    os.makedirs("figures", exist_ok=True)
    plot_reliability(centers, accs, counts, "figures/reliability-diagram.png")
    print("\nSaved: figures/reliability-diagram.png")