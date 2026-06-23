# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 1 | 5 | 20% |
| lexical_trigger | 3 | 4 | 75% |
| domain_shift | 2 | 4 | 50% |
| length_extreme | 5 | 6 | 83% |
| sarcasm | 1 | 6 | 17% |
| other | 5 | 5 | 100% |
| **Overall** | **17** | **30** | **56.7%** |

---

## Confirmed hypotheses

Two categories confirmed my hypotheses most strongly:

**Negation (20% accuracy):** The model failed on 4 out of 5 negation examples,
confirming the hypothesis that it ignores negation cues and anchors on sentiment
trigger words instead. Row 1 ("The latest update did not improve battery life at
all") was predicted positive, likely because "improve" dominated the decision.
Row 5 ("I can not say this app has been helpful at all") was also predicted
positive, suggesting the model read "helpful" without processing "can not".
The only correct prediction in this category was row 6 ("The app never freezes
on my device"), likely because "freezes" is a strong enough negative cue that
even without the negation the model hesitated — or the word "never" happened to
appear in training data in a positive context.

**Sarcasm (17% accuracy):** The model failed on 5 out of 6 sarcasm examples,
confirming the hypothesis that it classifies based on surface polarity rather
than intended meaning. Row 18 ("Oh great another update that broke everything
again") was predicted positive because "great" is a dominant positive trigger.
Row 19 ("Wow what an amazing app it only crashes five times a day") was also
predicted positive despite the obvious negative content following "amazing".
Row 29 ("Best decision I ever made installing this app said no one ever") was
the only correct prediction, likely because the explicit sarcasm marker
"said no one ever" shifted the probability enough toward negative.

---

## Refuted hypotheses

Two categories performed better than expected:

**length_extreme (83% accuracy):** I expected the model to struggle with very
short inputs due to lack of context, but it handled most cases correctly.
Row 13 ("Crashes always") was correctly predicted negative, and row 14
("Love it") was correctly predicted positive, suggesting the model has learned
strong associations between individual anchor words and sentiment labels even
without surrounding context. The one failure was row 15 ("Ok"), which was
predicted positive rather than neutral, likely because the training distribution
skewed toward binary labels and "Ok" appeared more often in positive contexts.

**lexical_trigger (75% accuracy):** I expected the model to be heavily misled
by strong sentiment words whose polarity is flipped by context, but it succeeded
on 3 out of 4 examples. Row 9 ("I loved this app until they ruined it with the
last update") was correctly predicted negative, suggesting the model does capture
some contextual reversal when the flipping phrase is explicit enough. The failure
was row 7 ("The app used to be fast but now it is painfully slow"), predicted
positive, which confirms that temporal cues like "used to be" are harder for the
model to process than direct negations.

---

## What the results reveal about the decision boundary

The adversarial results reveal two specific and consistent patterns in how the
model draws its decision boundary:

**1. The model is a lexical cue classifier, not a semantic reasoner.**
Across negation and sarcasm failures, the pattern is identical: the model
identifies the strongest sentiment word in the input and assigns the label that
word implies, regardless of whether a negation, a sarcasm marker, or a
contrastive conjunction modifies it. "Improve", "helpful", "great", and "amazing"
all pulled predictions toward positive even when the surrounding context
explicitly reversed their polarity. This means the decision boundary is drawn
primarily in lexical space, not in semantic or syntactic space.

**2. The model handles explicit signals better than implicit ones.**
Length extreme examples with single strong words ("Horrible", "Love it") were
classified correctly, and the one sarcasm example with an explicit marker
("said no one ever") was also correct. This suggests the model's boundary
responds well to unambiguous, high-frequency training signals but collapses
when the correct label requires resolving a conflict between surface form and
intended meaning. The practical implication is that the model would likely
perform poorly in any real-world deployment where users express negative
sentiment through irony, negation, or temporal contrast rather than direct
negative language.