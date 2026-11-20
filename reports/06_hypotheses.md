# Step 10. Hypothesis tests for the product team

Notebook: `notebooks/10_hypotheses.ipynb`

**What we did:** answered four business questions with statistics. For each question we report the conversion rate per group, the difference with a 95% confidence interval, and a two-proportion z-test.

**Why:** a model ranks visitors, but the business also asks direct questions like "is group A better than group B, and by how much?". These need a test, not a model.

**Two safety checks:**
1. **Client level.** Sessions of the same client are not independent, so we repeated every test with one row per client.
2. **Traffic mix.** Groups can differ because they come from different channels, so we compared devices *within* each channel.

## Results (1.73M sessions with events)

| Question | Group A | Group B | Difference | 95% CI | Relative | Client level |
|---|---|---|---|---|---|---|
| Organic vs paid | **4.44%** | 2.34% | +2.10 pp | [2.04, 2.17] | **+90%** | +96% ✓ |
| Mobile vs desktop | 2.80% | 3.32% | −0.51 pp | [−0.58, −0.45] | −16% | −14% ✓ |
| Moscow & SPb vs other cities | 2.99% | 2.78% | +0.20 pp | [0.15, 0.25] | +7% | +11% ✓ |
| Social ads vs other paid | **1.55%** | 2.53% | −0.98 pp | [−1.03, −0.92] | **−39%** | −43% ✓ |

All p-values are far below 0.001. The client-level results point the same way, so the conclusions are robust.

## Statistical significance is not importance

With 1.7 million sessions, almost any difference is "significant". So we judge the **size** of the effect:

- **Organic vs paid (+90%)** and **social ads (−39%)** are large. They are worth acting on.
- **Moscow & SPb (+7%, +0.2 pp)** is real but small. It is not a reason to change the strategy.

## Mobile vs desktop: the overall number is misleading

| Channel | Mobile | Desktop | Difference |
|---|---|---|---|
| banner | 2.88% | 2.22% | **mobile better** |
| cpm | 1.33% | 0.45% | **mobile better** |
| cpc | 2.36% | 2.51% | desktop better |
| (none) = direct | 3.54% | 4.55% | desktop better |
| referral | 5.66% | 6.36% | desktop better |

Overall, desktop looks better. But inside banner and CPM ads, mobile converts better. Part of the overall gap is **traffic mix**: desktop gets more direct and referral traffic, which converts well anyway.

This is a Simpson's-paradox-type effect, so the recommendation has to be made **per channel**, not "mobile is worse".

## Recommendations

1. **Invest in organic** (SEO, brand). It converts almost 2x better than paid.
2. **Review social-ads spend.** It converts 39% worse than other paid traffic. Check whether its cost per visit is low enough to justify that.
3. **Do not optimise for mobile vs desktop globally.** Decide per channel.
4. **Note:** these are observational comparisons, not experiments. They show *where* to look; an A/B test would prove the cause.
