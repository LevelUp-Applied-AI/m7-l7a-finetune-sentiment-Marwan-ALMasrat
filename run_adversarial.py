"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    
    predicted_labels = []
    predicted_probabilities = []
    correct_list = []

    for _, row in df.iterrows():
        # Tokenize
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Softmax + argmax
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0][pred_index].item()

        # Convert index to label via id2label
        pred_label = model.config.id2label[pred_index]

        # Check correctness
        is_correct = pred_label.lower() == str(row["expected_label"]).lower()

        predicted_labels.append(pred_label)
        predicted_probabilities.append(round(pred_prob, 4))
        correct_list.append(is_correct)

    df["predicted_label"] = predicted_labels
    df["predicted_probability"] = predicted_probabilities
    df["correct"] = correct_list

    return df


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")
    
    # Print summary
    total = len(df)
    correct = df["correct"].sum()
    print(f"Overall accuracy: {correct}/{total} ({correct/total:.1%})")
    
    print("\nPer-category accuracy:")
    for category, group in df.groupby("hypothesis_category"):
        cat_correct = group["correct"].sum()
        cat_total = len(group)
        print(f"  {category}: {cat_correct}/{cat_total}")


if __name__ == "__main__":
    main()