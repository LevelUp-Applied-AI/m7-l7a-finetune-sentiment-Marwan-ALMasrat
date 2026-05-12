"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8, temperature: float = 1.0):
    """
    Run manual PyTorch inference over a list of texts.

    Args:
        temperature: T > 1 lowers confidence (temperature scaling).
                     T = 1.0 means no scaling (default).

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            encoding = tokenizer(
                batch_texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            outputs = model(**encoding)
            logits  = outputs.logits

            # temperature scaling: divide logits by T before softmax
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    preds     = np.argmax(all_probs, axis=1)

    return preds, all_probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # only evaluate on non-abstained samples (pred != -1)
    mask   = y_pred != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    accuracy = float(np.sum(y_pred == y_true) / len(y_true)) if len(y_true) > 0 else 0.0

    classes   = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}

    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[int(c)] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "accuracy":  round(accuracy,  4),
        "macro_f1":  round(macro_f1,  4),
        "per_class": per_class,
    }


def predict_with_abstention(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 8,
    temperature: float = 1.0,
    threshold: float = 0.55,
    neutral_class: int = 1,
    neutral_threshold: float = 0.35,
):
    """
    Run inference with threshold-based abstention + neutral fix.

    Two abstention rules:
      1. General: if max(probs) < threshold           → abstain (-1)
      2. Neutral boost: if prob[neutral] >= neutral_threshold
                        AND neutral is not already the winner → force neutral

    The neutral boost catches cases where the model splits confidence
    between negative and positive but neutral has meaningful probability.

    Returns (preds, probs, abstained):
      preds:     shape (N,), int class indices (-1 = abstained)
      probs:     shape (N, num_classes)
      abstained: shape (N,), bool
    """
    preds, probs = manual_predict(model, tokenizer, texts, batch_size, temperature)

    # ── neutral boost ─────────────────────────────────────────────────────────
    neutral_prob    = probs[:, neutral_class]          # (N,)
    not_neutral_win = preds != neutral_class           # samples where neutral didn't win
    boost_mask      = (neutral_prob >= neutral_threshold) & not_neutral_win
    preds[boost_mask] = neutral_class

    # ── general abstention ────────────────────────────────────────────────────
    confidence = probs.max(axis=1)
    abstained  = confidence < threshold
    preds[abstained] = -1

    return preds, probs, abstained


if __name__ == "__main__":
    import json
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ── config ────────────────────────────────────────────────────────────────
    MODEL_DIR         = "model"
    TEMPERATURE       = 1.5    # T > 1 lowers over-confidence
    THRESHOLD         = 0.55   # general abstention (lowered from 0.70)
    NEUTRAL_THRESHOLD = 0.35   # neutral boost trigger

    # ── load ──────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    df       = pd.read_csv("predictions.csv")
    texts    = df["text"].tolist()
    label2id = {v: k for k, v in model.config.id2label.items()}
    y_true   = np.array([label2id[l] for l in df["label"].tolist()])

    # ── standard (no scaling, no abstention) ─────────────────────────────────
    print("=== Standard (no scaling, no abstention) ===")
    preds_base, probs_base = manual_predict(model, tokenizer, texts, temperature=1.0)
    report_base = compute_classification_report_from_arrays(y_true, preds_base)
    print(json.dumps(report_base, indent=2))

    # ── temperature scaling only ──────────────────────────────────────────────
    print(f"\n=== Temperature Scaling (T={TEMPERATURE}) ===")
    preds_t, probs_t = manual_predict(model, tokenizer, texts, temperature=TEMPERATURE)
    report_t = compute_classification_report_from_arrays(y_true, preds_t)
    print(json.dumps(report_t, indent=2))

    # ── abstention + neutral boost ────────────────────────────────────────────
    print(f"\n=== Abstention (T={TEMPERATURE}, threshold={THRESHOLD}, neutral_threshold={NEUTRAL_THRESHOLD}) ===")
    preds_a, probs_a, abstained = predict_with_abstention(
        model, tokenizer, texts,
        temperature=TEMPERATURE,
        threshold=THRESHOLD,
        neutral_threshold=NEUTRAL_THRESHOLD,
    )
    n_abstained = abstained.sum()
    print(f"Abstained : {n_abstained} / {len(texts)} samples ({n_abstained/len(texts)*100:.1f}%)")
    print(f"Answered  : {(~abstained).sum()} samples ({(~abstained).sum()/len(texts)*100:.1f}%)")

    report_a = compute_classification_report_from_arrays(y_true, preds_a)
    print("Metrics on non-abstained samples:")
    print(json.dumps(report_a, indent=2))

    # ── save probs for calibration.py ─────────────────────────────────────────
    np.save("manual_probs.npy",   probs_base)
    np.save("manual_probs_t.npy", probs_t)
    np.save("manual_ytrue.npy",   y_true)
    print("\nSaved: manual_probs.npy, manual_probs_t.npy, manual_ytrue.npy")