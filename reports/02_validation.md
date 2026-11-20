# Step 2. Validation design

Notebook: `notebooks/02_validation.ipynb`

## The question

How should we split the data into training and test sets so that the test score is **honest**: close to what the model would really achieve on new, future traffic?

The test score is only useful if it predicts real-world performance. A split that leaks information gives a beautiful number that falls apart in production.

## What we did

1. Built the modelling table: **1,732,266 sessions** after excluding the 127,776 sessions with no logged events (Step 1). Conversion rate: **2.90%**.
2. Trained **the same simple model** (LightGBM on session features: traffic source, campaign, device, city, visit number, hour, weekday) under **three different splits**.
3. Compared the test scores. Only the split changed, so any difference in the score comes from the split itself.

## The three splits

**A. Random split (the most common default).** 80% of sessions go to training, 20% to test, at random.
Problem: **32% of test sessions belong to visitors who also appear in training**, and the test covers the same months as training.

**B. Random split by visitor.** Each visitor goes entirely into training or entirely into test. This removes visitor overlap but still mixes all months.

**C. Time split.** Train on the past, test on the future:

| Set | Period | Sessions | Purpose |
|---|---|---|---|
| Train | May – October | 1,102,186 | fit the model |
| Validation | November | 263,835 | tuning, early stopping, choosing the threshold |
| Test | December | 366,245 | final score, used **once**, at the very end |

## Results

| Split | ROC-AUC | Test conversion rate |
|---|---|---|
| A. Random split | **0.705** | 2.90% |
| B. Random split by visitor | 0.701 | 2.95% |
| C. Time split (test = December) | **0.612** | 1.99% |

**The same model loses almost 0.10 of ROC-AUC when it is tested on the future.** The random split was hiding this.

## Why the score drops: the root cause

Visitor overlap explains only a small part (0.705 to 0.701). The main reason is **time**. I checked what December traffic looks like compared to the training months:

| Feature | Share of December sessions with a value never seen in May–October |
|---|---|
| utm_adcontent (ad creative) | **45%** |
| utm_campaign | **25%** |
| utm_source | 9% |
| geo_city | 0.1% |

**A quarter of December traffic comes from campaigns that did not exist during training, and almost half from new ad creatives.** The model had memorised which *specific* campaign IDs converted well. When marketing launched new campaigns, that knowledge became useless.

On top of that, the conversion rate itself fell from 2.9% to 2.0%.

## Decision

- **Time split: train May–October, validate on November, test on December.**
- The December test set is touched only once, at the end. All decisions are made on November.
- Returning visitors in December are allowed: in production we *would* know their past visits. What we must guarantee is that **every feature uses only information available before the session starts**.

## What this means for the next steps

The 0.61 is now our **honest baseline**. To improve it, the model needs features that **generalise to new campaigns** instead of memorising campaign IDs, for example the type of traffic (medium), whether a campaign is new, and the visitor's own history. That is the focus of Step 5.
