# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram shows a consistent pattern: every bucket bar sits **below** the perfect-calibration diagonal, meaning the model is **under-confident** across all confidence ranges.

Specific bucket values from the run:

| Bucket center | Empirical accuracy | Count |
|:---:|:---:|:---:|
| 0.45 | 0.3647 | 85 |
| 0.55 | 0.4672 | 244 |
| 0.65 | 0.5000 | 260 |
| 0.75 | 0.6220 | 254 |
| 0.85 | 0.7170 | 311 |
| 0.95 | 0.8614 | 339 |

The 0.85–0.95 confidence bucket showed accuracy of only 0.717, indicating the model is over-confident in this range. The 0.55–0.65 buckets are the weakest, with accuracy of 0.467–0.500 despite confidence of 0.55–0.65. No predictions were made below 0.35 confidence — the model always commits to a class with at least 40% probability.

## Expected Calibration Error

**ECE = 0.1146**

An ECE of 0.1146 means that on average, the model's stated confidence deviates from its actual accuracy by about **11.5 percentage points**. A well-calibrated model typically achieves ECE below 0.05. At 0.1146, the probability scores from this model should not be used directly as trustworthy confidence estimates in production — for example, using a 0.80 confidence threshold to trigger an automated action would fire more often than expected, because the model's 0.80 confidence corresponds to roughly 0.62–0.72 actual accuracy.

## A specific calibration pattern

The most notable pattern is **over-confidence in the mid-range buckets (0.55–0.65)**, where accuracy is only 0.467–0.500 despite confidence of 0.55–0.65. This arises directly from the weakness of the neutral class (class 1), which achieved precision of 0.471, recall of 0.521, and F1 of 0.494. When the model is uncertain between negative and positive, it picks one with moderate softmax confidence rather than routing through neutral, pushing many ambiguous samples into the mid-range confidence buckets with lower-than-expected accuracy. This pattern arose because DistilBERT was fine-tuned for only 2 epochs on a 3-class problem where neutral is inherently ambiguous, leaving the model under-trained on that decision boundary.

## A proposed engineering action

Add **label smoothing** during training by setting `label_smoothing_factor=0.1` in `TrainingArguments` inside `lab.py`. Label smoothing prevents the model from becoming over-confident during training by replacing hard targets (0 or 1) with soft targets (0.05 or 0.95), which directly reduces ECE without requiring any separate calibration set or post-hoc correction. Unlike temperature scaling, label smoothing has no data leakage risk because it operates entirely during the training phase. As a complementary action, adding `class_weight="balanced"` would give the neutral class a higher loss weight proportional to its underrepresentation, addressing both the calibration gap in the mid-range buckets and the weak neutral F1 simultaneously.