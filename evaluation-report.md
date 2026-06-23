# Module 7 Week A — Lab Evaluation Report

## Dataset

The AARSynth app reviews dataset contains 7,472 labeled reviews across 9 apps with three sentiment classes: negative (0), neutral (1), and positive (2). The dataset was split into 80% training (~5,977 examples) and 20% test (~1,495 examples) using a fixed seed of 42.

## Model and Hyperparameters

- **Backbone:** distilbert-base-uncased
- **Number of labels:** 3 (negative, neutral, positive)
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8 (train and eval)
- **Max length:** 128 tokens
- **Seed:** 42
- **Training time:** ~56 minutes on CPU

## Metrics on the Test Split

**Aggregate:**

| Metric | Value |
|---|---|
| Accuracy | 0.6341 |
| Macro-F1 | 0.6326 |

**Per class:**

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Negative | 0.7228 | 0.7143 | 0.7315 |
| Neutral  | 0.4944 | 0.4707 | 0.5205 |
| Positive | 0.6806 | 0.7246 | 0.6417 |

## Confusion Matrix

|  | Predicted: Negative | Predicted: Neutral | Predicted: Positive |
|---|---|---|---|
| **True: Negative** | 365 | 118 | 16 |
| **True: Neutral**  | 108 | 241 | 114 |
| **True: Positive** | 38  | 153 | 342 |

The model performs best on the **negative** class (F1: 0.72) and struggles most with **neutral** (F1: 0.49). The most common confusion is between neutral and the two polar classes — 108 neutral examples were predicted negative, and 114 were predicted positive. This is expected: neutral reviews contain mixed or mild sentiment cues that pull toward either extreme.

## Three Qualitative Error Examples

> **Note:** Replace the sentences below with actual rows from `predictions.csv` where `label != predicted_label`.

**Example 1 — True: Neutral, Predicted: Negative**

- **Sentence:** *(Find a row in predictions.csv where label=neutral and predicted_label=negative)*
- **Gold label:** neutral
- **Predicted label:** negative
- **Gold-class probability:** *(read prob_neutral for that row)*
- **Analysis:** Neutral reviews that contain complaint-like language ("could be better", "not great") or negation words likely trigger the negative class. The model may overweight individual negative cue words without considering the overall moderate tone of the review.

---

**Example 2 — True: Positive, Predicted: Neutral**

- **Sentence:** *(Find a row in predictions.csv where label=positive and predicted_label=neutral)*
- **Gold label:** positive
- **Predicted label:** neutral
- **Gold-class probability:** *(read prob_positive for that row)*
- **Analysis:** Long positive reviews that also mention minor issues may confuse the model. If a review says "great app but could use a few improvements," the model may average the sentiment signals and land on neutral rather than recognizing the dominant positive tone.

---

**Example 3 — True: Negative, Predicted: Neutral**

- **Sentence:** *(Find a row in predictions.csv where label=negative and predicted_label=neutral)*
- **Gold label:** negative
- **Predicted label:** neutral
- **Gold-class probability:** *(read prob_negative for that row)*
- **Analysis:** Negative reviews expressed in polite or understated language ("I wish it worked better", "not what I expected") may lack the strong negative cue words the model relies on, causing it to predict neutral. This is a known failure mode for sentiment models trained on explicit opinions.

## Hugging Face Hub Model URL

https://huggingface.co/MrMarwans/m7-app-review-sentiment