# Steps 6–8. More features, other models, two use cases

Notebooks: `notebooks/06_more_features.ipynb`, `07_models.ipynb`, `08_final_test.ipynb`

## Final results (December test, used once)

| Model | Use case | December ROC-AUC | Top 10% lift | Top 30% share of conversions |
|---|---|---|---|---|
| Single LightGBM | session start | 0.677 | 2.6x | 53% |
| **Blend of LightGBM + XGBoost + CatBoost** | **session start** | **0.685** | **2.8x** | **55%** |
| **LightGBM with first 3 events** | **early session** | **0.780** | **3.7x** | **71%** |

Honest baseline for comparison: **0.612** (raw attributes, time split).

## Two use cases, two models

The business can use a conversion score at two different moments:

1. **Session start**: the moment a visitor lands, before they have done anything. Used for **marketing decisions**: which campaigns and audiences bring visitors likely to convert, where to spend the budget.
2. **Early session**: after the visitor's first 3 events. Used for **on-site actions**: show a chat invitation, offer a callback, or highlight a promotion for visitors who look likely to convert.

They are different problems with different information, so they get different models.

## Step 6. Three new feature ideas

All still known when the session starts.

**1. Recent conversion rates (last 7 and 30 days)** per source, medium, campaign, creative and city, plus the overall site conversion rate over the last 7 days.
*Idea:* traffic quality drifts, so a campaign's rate since May can be stale. A rolling window reacts to drift, and a new campaign gets a real rate one day after launch.

**2. What the visitor did on previous visits**: how many car cards they viewed, searches, photo views, catalogue browsing, quiz, chat, offer clicks. Previous visits are finished before the current one starts.

**3. Landing URL parameters**: whether the click came from Yandex or Facebook ads, whether the link had a city or price filter, whether it was a rental page, how many parameters the link had.

## Step 7. Experiments on November

### New features (LightGBM)

| Experiment | Features | November ROC-AUC |
|---|---|---|
| F0 previous best (E5) | 48 | 0.679 |
| F1 + recent conversion rates | 69 | 0.677 |
| F2 + previous-visit behaviour | 77 | 0.677 |
| F3 + landing URL parameters | 83 | 0.679 |

**None of the new features helped.** The information was already in the model: the cumulative conversion rate and the age of each campaign already describe new campaigns, and visitor history was already captured by previous visits and conversions. This is a **ceiling of information** for the session-start problem, not a modelling failure.

### Weighting recent data

Sessions weighted by recency (weight halves every N days):

| Half-life | November ROC-AUC |
|---|---|
| 30 days | 0.674 |
| 60 days | 0.676 |
| 120 days | 0.677 |
| no weighting | 0.679 |

**Weighting made it worse.** Older months still carry useful patterns; down-weighting them throws data away.

### Other models (same features, F3)

| Model | November ROC-AUC |
|---|---|
| LightGBM | 0.679 |
| XGBoost | 0.680 |
| CatBoost | 0.682 |
| **Blend (rank average of all three)** | **0.687** |

CatBoost is slightly better on its own, it has a built-in, leakage-safe way of encoding categories (ordered target statistics). The blend gains another 0.005: the three models make **different mistakes**, and averaging their rankings cancels part of them.

Rank averaging is used instead of averaging probabilities because the models' probabilities are on different scales; ranks put them on the same scale.

### Early-session model

Sessions where a target action happens within the first 3 events are excluded, there is nothing left to predict (about 0.6% of sessions).

| Features | November ROC-AUC |
|---|---|
| Session-start features only (same sessions) | 0.705 |
| **Session-start + first 3 events** | **0.791** |

Three events, opening a car card, using the search, viewing photos, already reveal intent. **This is the largest single gain in the whole project: +0.09.**

## Step 8. Final December test

Models retrained on May–November with the November best number of trees (+20% for the larger training set), then scored on December **once**.

### Session-start blend

| Model | December ROC-AUC | PR-AUC |
|---|---|---|
| LightGBM | 0.677 | 0.062 |
| XGBoost | 0.680 | 0.063 |
| CatBoost | 0.681 | 0.062 |
| **Blend** | **0.685** | **0.065** |

Random guessing gives PR-AUC 0.020, the blend is **3.3x better** at finding converters.

| Decile | Conversion rate | Lift | Cumulative share of conversions |
|---|---|---|---|
| 1 (top 10%) | **5.56%** | **2.8x** | 27.9% |
| 2 | 2.79% | 1.4x | 41.9% |
| 3 | 2.51% | 1.3x | **54.5%** |
| 10 (bottom 10%) | 0.37% | 0.2x | 100% |

### Early-session model

| | December |
|---|---|
| ROC-AUC | **0.780** |
| PR-AUC | 0.069 (random 0.016): **4.4x better than random** |

| Decile | Conversion rate | Lift | Cumulative share of conversions |
|---|---|---|---|
| 1 (top 10%) | **5.83%** | **3.7x** | 37.0% |
| 2 | 3.00% | 1.9x | 56.0% |
| 3 | 2.29% | 1.5x | **70.5%** |
| 10 (bottom 10%) | 0.13% | 0.1x | 100% |

**After just 3 events, the top 30% of visitors contain 71% of all future conversions.** If the site offers a chat or a callback only to that 30%, it reaches most of the people who are going to convert, without disturbing the other 70%.

## Stability: November vs December

| Model | November | December |
|---|---|---|
| Session-start blend | 0.687 | 0.685 |
| Early-session | 0.791 | 0.780 |

The scores barely move between validation and test, so the models were **not overfitted to November**, and they hold up on new traffic.
