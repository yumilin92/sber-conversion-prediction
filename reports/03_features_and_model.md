# Steps 3–5. Features, experiments and the final model

Notebooks: `notebooks/03_features.ipynb`, `04_experiments.ipynb`, `05_final_model.ipynb`

## Result in one line

**December test ROC-AUC: 0.612 to 0.677.** The top 10% of sessions ranked by the model convert **2.6x better** than average, and the top 30% contain **53% of all conversions**.

## The rule for every feature

A feature is allowed only if it is **known at the moment the session starts**. Anything that happens later in the session, clicks, pages after landing, would leak the answer, because the target itself is a click inside the session.

## Step 3. Five feature groups

**1. Raw session attributes.** Traffic source, medium, campaign, ad creative, keyword, device, browser, city, visit number, hour, weekday.

**2. Generalising traffic features.** These describe *what kind* of traffic a session is, so they still work for campaigns the model has never seen:
- is the traffic organic; is it social-media advertising; is the city one where the service operates (Moscow, Saint Petersburg); does the visit have a campaign or a keyword;
- **campaign age**: how many days a campaign, creative or source has existed when the session starts. A brand-new campaign gets 0;
- screen width, height and area.

**3. Historical conversion rates (time-aware target encoding).** For each source, medium, campaign, creative, keyword, city, device brand and browser: its conversion rate **using only days strictly before the session**.
- Rates are **smoothed** towards the overall rate (with a weight of 200 sessions), so a category with 3 past visits doesn't get 0% or 100%.
- We also keep the number of past sessions, so the model knows how reliable each rate is.
- Sessions without logged events are excluded from these statistics, because their label is unknown.

**4. Visitor history.** From the visitor's **previous** sessions, which are finished before the current one starts: number of previous visits, number of previous conversions, whether they converted before, days since the last visit, how many events the previous visit had, and the medium of their very first visit.

**5. Landing page.** The first page of the visit is known as soon as the visitor arrives: the section (home page, car catalogue, FAQ, promo page), the car brand if they landed on a car page, and the page depth.

## Step 4. Experiments

Train on May–October, evaluate on **November**. Feature groups are added one at a time, so the contribution of each is visible. December is not used at all.

| Experiment | Features | November ROC-AUC |
|---|---|---|
| E0 raw session attributes | 14 | 0.648 |
| E1 + generalising traffic features | 26 | 0.643 |
| E2 + historical conversion rates | 42 | 0.647 |
| E3 + visitor history | 49 | 0.656 |
| E4 + landing page | 52 | 0.658 |
| **E5 = E4 without raw campaign / ad IDs** | **48** | **0.675** |
| E6 = E5 without raw city / brand / browser | 45 | 0.674 |
| E7 = E5 without past-session counts | 40 | 0.673 |

**The biggest gain came from *removing* features.** As long as the raw campaign and ad IDs were in the model, it kept memorising them, and the useful new features were drowned out. Once the IDs were removed, the model had to rely on features that describe campaigns in a general way, their past conversion rate and their age, and the score jumped by 0.017.

This confirms the diagnosis from Step 2: a quarter of future traffic comes from campaigns the model has never seen, so memorising IDs doesn't transfer.

### Tuning tree size (on E5)

| num_leaves | min_child_samples | November ROC-AUC |
|---|---|---|
| 15 | 1000 | 0.671 |
| 31 | 500 | 0.673 |
| 63 | 200 | 0.675 |
| **127** | **200** | **0.679** |

Tuning adds little compared with the feature work. That is typical: features matter more than hyperparameters.

**Model:** LightGBM with native categorical handling, early stopping on November.
**Categories:** learned from training data only; a value never seen in training is treated as missing.

## Step 5. Final model and the December test

The final model was retrained on **May–November** (the November best iteration was 128 trees; I used 155 because the training set is bigger) and scored on **December once**.

| | ROC-AUC on December |
|---|---|
| Raw session attributes, random split (looks good, but misleading) | 0.705 (on a random test set) |
| Raw session attributes, honest time split | 0.612 |
| **Final model, honest time split** | **0.677** |

PR-AUC: **0.062** vs **0.020** for random guessing, about **3x better** than random at finding converters.

### What the score means for the business

December sessions ranked by predicted probability and split into 10 equal groups:

| Decile | Conversion rate | Lift | Share of all conversions (cumulative) |
|---|---|---|---|
| 1 (top 10%) | **5.14%** | **2.6x** | 25.8% |
| 2 | 3.02% | 1.5x | 40.9% |
| 3 | 2.35% | 1.2x | **52.7%** |
| 5 | 1.99% | 1.0x | 73.4% |
| 10 (bottom 10%) | 0.42% | 0.2x | 100% |

- The top 10% of sessions convert at 5.1%: **2.6x the average** of 2.0%.
- The **top 30% of sessions contain 53% of all conversions**.
- The bottom 10% convert at only 0.4%.

**How this would be used:** spend budget on campaigns and audiences whose sessions score high, trigger a chat or callback offer for high-scoring visitors, and cut spend on traffic that consistently scores at the bottom.

### What drives the prediction (gain importance)

| Feature | Share of importance |
|---|---|
| prev_conversions, visitor converted before | 15.2% |
| utm_campaign_past_cr, campaign's past conversion rate | 13.6% |
| geo_city | 6.5% |
| utm_source_past_cr, source's past conversion rate | 4.9% |
| first_medium, how the visitor first arrived | 3.4% |

Two things matter most: **who the visitor is** (their history) and **how good their campaign has been so far**.

## Context for a score of 0.68

Predicting a purchase-intent click from **only what is known when a visit starts**: before the person has done anything, is hard. The answer is mostly decided by what happens *inside* the visit, which we deliberately don't use. ROC-AUC 0.68 on future data is realistic for this task, and the decile table shows it is useful: it separates a 5% converting group from a 0.4% one.

A score of 0.95+ on this kind of task almost always means leakage.
