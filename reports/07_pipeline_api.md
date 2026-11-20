# Step 11. From notebook to a working service

Code: `src/`, tests: `tests/`

**What we did:** turned the two models into a service that works from raw website data, checked that production features match the training features, added drift monitoring, and packaged everything in Docker with CI.

**Why:** a model in a notebook creates no value. It is useful only when other systems can call it, and only safe when someone notices that the traffic has changed.

## Two use cases, two ways of serving

| | Session-start blend | Early-session model |
|---|---|---|
| Who uses it | marketing team | the website |
| When | once a day | in real time, after 3 events |
| How | batch job, writes a table of scores | API, one answer per visitor |
| Output | score and decile (1 = top 10%) | probability and a `priority` flag (top 30%) |

The blend runs only in batch. It averages ranks, and ranks only make sense inside a batch of visits.

## Features from raw visits

The website does not know a campaign's past conversion rate or a visitor's history. It sends what it has: traffic source, device, city, visitor ID, landing page and the first events. The service builds the 90 model features itself, from two stores:

| Store | Contents | Updated |
|---|---|---|
| category stats (`store/category_stats.parquet`) | past and recent (7 and 30 days) conversion rate and volume of every source, medium, campaign, creative, keyword, city, device brand and browser; first-seen date of each campaign | daily, by `src/build_store.py` |
| visitor history (`store/clients.sqlite`) | number of previous visits and conversions, last visit time, events on previous visits | daily rebuild, plus `POST /visits/finished` after every visit |

Because the API records each finished visit, a second visit on the same day already sees the first one, exactly as in training.

`src/features.py` holds all feature logic. The API and the batch job both use it, so the two cannot drift apart.

## Proof: production features = training features

`tests/test_parity.py` takes a full December day (6,593 visits) and compares three paths:

1. features built by `src/features.py` against the features the model was trained on: all 90 match. The only differences are 2 visits where one client had two visits in the same second, whose order is arbitrary;
2. model scores from both: identical;
3. the API, replaying 300 visits one by one in time order, as the website would: the same probabilities as the batch job.

This test found two bugs before they could reach production: an empty car brand turned into the text `'nan'` instead of `(none)`, and new visitors broke the data types. Both would have lowered the quality silently. This kind of problem is called **training-serving skew**.

The test needs the original data, so it runs locally and is skipped in CI. The other 22 tests use synthetic data.

## Monitoring

`python -m src.monitor report --day 2021-12-15` compares one day with November, the last training month:

- **PSI of the model's probability**: below 0.1 stable, 0.1 to 0.25 worth a look, above 0.25 investigate;
- **PSI of behaviour features** (visitor history, landing page, first events);
- **traffic mix**: PSI of the shares of channels, devices, cities and landing sections;
- **share of unseen categories**;
- **conversion rate and ROC-AUC**, once the day's outcomes are known.

Features that change with the calendar by design (weekday, campaign age, historical rates) are left out, because a single day always looks different from a month.

| Day | Status | Probability PSI | Largest shift | ROC-AUC |
|---|---|---|---|---|
| 25 Nov | ok | 0.01 | channel mix 0.10 | 0.86 |
| 15 Dec | alert | 0.19 | landing car brand 1.10 | 0.75 |
| 30 Dec | alert | 0.25 | channel mix 0.52 | 0.82 |

The December alerts match what the data audit showed: new campaigns, different landing pages and falling conversion. In production this would trigger a look at the traffic and, if needed, retraining.

## The pieces

| File | Role |
|---|---|
| `src/features.py` | all feature logic, shared by batch and API |
| `src/build_store.py` | daily job: rebuild both stores |
| `src/batch_score.py` | daily job: score a day's visits with the blend |
| `src/api.py` | FastAPI: `/score/early-session`, `/visits/finished`, `/health` |
| `src/monitor.py` | daily drift and quality report |
| `models/model_card.json` | version, training period, test metrics, threshold rule |

The API validates every request, rejects batches over 500 visits, returns 503 if the store is missing, reports a stale store in `/health` and logs every request with its latency.

The priority threshold (top 30%) is the 70th percentile of recent scores. It needs no outcomes, so it can be refreshed from live traffic.

## Tests

| Area | What is checked |
|---|---|
| features | no leakage from later days or from today's outcomes; new visitors; earlier visits on the same day; unseen campaigns; landing page parsing |
| scoring | valid scores and row order; same input gives the same output; missing features fail loudly; unseen categories; more car-card views give a higher score |
| API | raw visit in, score out; a finished visit becomes history; visits that converted in the first events are not scored; bad input (422), oversized batch (413), missing store (503) |
| monitoring | PSI is 0 for the same data and grows with the shift; missing values get their own bin |
| parity | the three-way comparison above, on real data |

## Limitations

- The data ends in December 2021, so the store and the model describe that period.
- SQLite is enough for one server. With more traffic, the visitor history would move to Redis or a database table behind the same interface.
- Retraining is manual: the notebooks produce the model files. A scheduled retraining job is the next step.
