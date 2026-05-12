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

The 0.85–0.95 confidence bucket showed accuracy of only 0.717, indicating the model is over-confident in this range. Similarly, the 0.95–1.00 bucket showed accuracy of 0.861 against a confidence of ~0.95, again falling short of the diagonal. No predictions were made below 0.35 confidence — the model always commits to a class with at least 40% probability.

## Expected Calibration Error

**ECE = 0.1146**

An ECE of 0.1146 means that on average, the model's stated confidence deviates from its actual accuracy by about **11.5 percentage points**. A well-calibrated model typically achieves ECE below 0.05. At 0.1146, the probability scores from this model should not be used directly as trustworthy confidence estimates in production — for example, using a 0.80 threshold to trigger an automated action would fire more often than expected because the model's 0.80 confidence corresponds to roughly 0.62–0.72 actual accuracy.

## A specific calibration pattern

The most notable pattern is **over-confidence in the mid-range buckets (0.55–0.65)**, where accuracy is only 0.467–0.500 despite confidence of 0.55–0.65. This arises directly from the collapse of the neutral class (class 1): the model achieves precision of 0.471 and recall of 0.521 on neutral, meaning it struggles at the boundary between negative and positive. When the model is uncertain between the two polar classes, it picks one with moderate softmax confidence rather than routing through neutral. This pushes many ambiguous samples into the mid-range confidence buckets with lower-than-expected accuracy. This pattern arose because DistilBERT was fine-tuned for only 2 epochs on a 3-class problem where neutral is inherently ambiguous, leaving the model under-trained on that decision boundary.

## A proposed engineering action

Apply **temperature scaling** before deployment. Temperature scaling divides the logits by a learned scalar T > 1 before softmax, which lowers over-confident scores without changing the model's predictions. It requires only a small held-out calibration set and would bring the mid-range buckets closer to the diagonal. As a complementary action, add **threshold-based abstention**: if `max(probs) < 0.70`, route the review to a human reviewer rather than auto-labeling it. This directly addresses the weak neutral boundary where the model is least reliable, reducing the risk of silent mislabeling in production.