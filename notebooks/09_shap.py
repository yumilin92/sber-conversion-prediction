# %% [markdown]
# # 09. Explaining the models with SHAP
#
# **What:** for every prediction, SHAP splits the score into a contribution from each feature.
# **Why:** the business needs to know *why* a visitor scores high, and we need to check the model
# learned sensible patterns, not artefacts.
#
# We explain the LightGBM part of the session-start blend and the early-session model on December data.

# %%
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROCESSED = "../data/processed"
FIG = "../reports/figures"
df = pd.read_parquet(f"{PROCESSED}/features_v2.parquet")
start = joblib.load("../models/session_start_blend.joblib")
early = joblib.load("../models/early_session_lgbm.joblib")
CATS = ["utm_medium", "device_category", "device_os", "device_brand", "device_browser",
        "geo_country", "geo_city", "first_medium", "landing_section", "landing_car_brand"]

def december(frame, features):
    fit = frame[frame["visit_dt"] < "2021-12-01"]
    test = frame[frame["visit_dt"] >= "2021-12-01"].copy()
    for c in CATS:
        if c in features:
            test[c] = test[c].astype(pd.CategoricalDtype(fit[c].unique()))
    return test.sample(100_000, random_state=42)[features]

def explain(model, X, name):
    contrib = model.predict(X, pred_contrib=True)[:, :-1]          # last column = base value
    imp = pd.Series(np.abs(contrib).mean(0), index=X.columns).sort_values(ascending=False)
    Xplot = X.apply(lambda s: s.cat.codes if s.dtype.name == "category" else s)
    shap.summary_plot(contrib, Xplot, max_display=12, show=False, plot_size=(8, 6))
    plt.title(name); plt.tight_layout(); plt.savefig(f"{FIG}/shap_{name}.png", dpi=130); plt.close()
    return imp, contrib

# %% [markdown]
# ## 1. Session-start model

# %%
X1 = december(df, start["features"])
imp1, c1 = explain(start["lgb"], X1, "session_start")
(imp1 / imp1.sum() * 100).head(12).round(1)

# %% [markdown]
# ## 2. Early-session model

# %%
X2 = december(df[df["target_in_first3"] == 0], early["features"])
imp2, c2 = explain(early["model"], X2, "early_session")
(imp2 / imp2.sum() * 100).head(12).round(1)

# %% [markdown]
# ## 3. Direction: does a higher value push the score up or down?

# %%
def direction(X, contrib, top):
    rows = []
    for f in top:
        i = list(X.columns).index(f)
        x = X[f]
        if x.dtype.name == "category":
            rows.append((f, "categorical")); continue
        ok = x.notna()
        r = np.corrcoef(x[ok].astype(float), contrib[ok.values, i])[0, 1]
        rows.append((f, "higher -> more likely" if r > 0 else "higher -> less likely"))
    return pd.DataFrame(rows, columns=["feature", "effect"])

print(direction(X1, c1, imp1.index[:10]))
print(direction(X2, c2, imp2.index[:10]))
