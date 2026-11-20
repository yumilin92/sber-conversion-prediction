# Step 1. Data audit

Notebook: `notebooks/01_data_audit.ipynb`

## What we did

1. Loaded both raw tables once and saved them as Parquet. The raw CSVs are 0.4 GB (sessions) and 4.3 GB (hits). Parquet is columnar and compressed, so later steps read only the columns they need, in seconds.
2. Checked the grain of each table: what one row means.
3. Checked whether the two tables match each other.
4. Measured missing values, including Google Analytics placeholders.
5. Built the target.
6. Looked at conversion over time before deciding how to split the data.

## 1. Grain: what one row means

| Table | One row = | Count |
|---|---|---|
| sessions | one **visit** to the website | **1,860,042** visits |
| hits | one **event** inside a visit (a click, a page view) | **15.7 million** events, about 9 per visit |
| client_id | one **visitor** (one browser) | **1.39 million** visitors |

Period: **19 May – 31 December 2021**.

Most visitors came once, and 90% came at most twice. But one client_id has **462 visits**: most likely a bot or a shared computer. Outliers like this need to be kept in mind.

**Why it matters:** the same person can visit many times. If their visits end up in both the training set and the test set, the model "recognises" them and the test score is inflated. We handle this in Step 2.

## 2. Finding #1: 7% of sessions have no events

**127,776 sessions (6.9%) have no events at all** in the hits table.

The easy shortcut would be to label them 0 ("did not convert"). But we actually **don't know** whether they converted, their events simply weren't recorded. I checked two things:

- it happens **across every channel** (banner, cpc, referral, direct);
- the share grows over time: from **3% in May to 11% in November**.

That pattern points to **a logging problem**, not to user behaviour.

**Decision:** exclude these sessions from modelling. Labelling them as "no conversion" would train the model on wrong answers.

## 3. Finding #2: Conversion falls over time

| Period | Weekly conversion rate |
|---|---|
| late May – June | **5–6%** |
| July – October | about 2.5–3.5% |
| December | **about 1.7–2%** |

Right after launch, the most interested people came first. Later, paid traffic scaled up (almost 3x more visits per week), and the average quality of traffic went down.

**Why this is critical for the model:** with a random split, the test set contains visits from the same months as the training set. In reality, a model always predicts **the future**, where the conversion rate is different. So the test set must come from **the most recent months**.

## 4. Missing values

| Field | Missing | What we do |
|---|---|---|
| device_model | 99% | drop, no information |
| utm_keyword | 58% | keep "empty" as its own category, it means the visit didn't come from search ads |
| device_os | 58% | same, empty as a category |
| utm_campaign, utm_adcontent | 12–18% | empty as a category |
| geo_city | 4.2% "(not set)" | Google Analytics writes "(not set)" instead of a blank, so a plain NaN check misses it |

**Key idea: a missing value is not always noise.** If a visit has no campaign, it is most likely organic traffic. So an empty value becomes a separate category instead of being deleted.

## 5. Target

A session counts as a **conversion** if it contains at least one of 8 target actions (leave a request, order a callback, etc.). Several target actions in one visit still count as one conversion.

Most frequent target actions:

| Action | Count |
|---|---|
| sub_car_claim_click ("subscribe" button) | 37,928 |
| sub_open_dialog_click (open dialog) | 25,870 |
| sub_submit_success (request submitted) | 18,439 |

**Conversion rate: 2.70%** (50,314 of 1.86 million sessions): a highly imbalanced problem.
