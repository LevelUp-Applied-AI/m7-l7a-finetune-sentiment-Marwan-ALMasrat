"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

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

            probs = torch.softmax(logits, dim=-1)
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

    accuracy = float(np.sum(y_pred == y_true) / len(y_true))

    # جميع الكلاسات الممكنة من y_true فقط (وليس y_pred)
    # حتى يظهر neutral حتى لو النموذج لم يتنبأ به
    classes = np.unique(y_true)
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


if __name__ == "__main__":
    import json
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MODEL_DIR = "model"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    df       = pd.read_csv("predictions.csv")
    texts    = df["text"].tolist()
    label2id = {v: k for k, v in model.config.id2label.items()}
    y_true   = np.array([label2id[l] for l in df["label"].tolist()])

    preds, probs = manual_predict(model, tokenizer, texts)

    report = compute_classification_report_from_arrays(y_true, preds)
    print(json.dumps(report, indent=2))

    np.save("manual_probs.npy", probs)
    np.save("manual_ytrue.npy", y_true)
    print("\nSaved: manual_probs.npy, manual_ytrue.npy")