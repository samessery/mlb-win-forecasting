"""
Can last season's underlying performance and payroll forecast next season's wins?

Setup
  * Unit: a team's season t, used to predict that same team's win percentage in t+1.
  * Data: 30 teams, 2015 to 2025 (FanGraphs / Statcast team metrics, Opening Day payroll).
  * 2020 (60 games) is dropped as both a predictor season and a target, so the pairs
    are 2015->16, 16->17, 17->18, 18->19 and 2021->22, 22->23, 23->24, 24->25
    (240 team-season pairs).
  * Evaluation is a rolling origin: for each target season, models are fit only on
    pairs whose target season is earlier. Targets scored: 2019, 2022, 2023, 2024, 2025
    (150 out-of-sample forecasts). No tuning on the test years.

Feature sets
  A  win% only                     (persistence)
  B  win% + log payroll
  C  skill: xwOBA, ISO, xERA, FIP, DEF   (the capstone predictors)
  D  C + log payroll + win%
  E  everything numeric (22 variables)

Run:  python python/forecast.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED = 488
TEST_YEARS = [2019, 2022, 2023, 2024, 2025]

SKILL = ["xwOBA", "ISO", "xERA", "FIP", "DEF"]
FEATURE_SETS = {
    "A win% only": ["WinPct"],
    "B win% + payroll": ["WinPct", "lnPayroll"],
    "C skill metrics": SKILL,
    "D skill + payroll + win%": SKILL + ["lnPayroll", "WinPct"],
}


def load_pairs():
    d = pd.read_csv(ROOT / "data" / "team_season_2015_2025.csv")
    d = d[d.Season != 2020].copy()
    nxt = d[["Team", "Season", "WinPct"]].copy()
    nxt["Season"] -= 1
    nxt = nxt.rename(columns={"WinPct": "target"})
    d = d.merge(nxt, on=["Team", "Season"], how="inner")      # season t features -> t+1 target
    d["target_season"] = d.Season + 1
    # 2020 rows were removed above, so no 2019->2020 or 2020->2021 pair can form.
    return d.reset_index(drop=True)


def all_numeric(d):
    drop = {"Season", "Team", "target", "target_season", "TeamID", "Wins", "Games", "Payroll"}
    cols = [c for c in d.columns if c not in drop and pd.api.types.is_numeric_dtype(d[c])]
    return [c for c in cols if d[c].notna().all()]       # drops UZR150 / OAA (missing in some seasons)


def models(seed=SEED):
    return {
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=seed, max_iter=20000)),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=8, max_features=0.5,
                                              random_state=seed, n_jobs=-1),
        "GradBoost": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                               subsample=0.8, random_state=seed),
    }


def rolling_forecast(d, feature_sets):
    rows = []
    for ty in TEST_YEARS:
        tr, te = d[d.target_season < ty], d[d.target_season == ty]
        # benchmarks that need no features
        rows.append(pd.DataFrame({"Team": te.Team.values, "target_season": ty, "y": te.target.values,
                                  "model": "Benchmark: 0.500", "features": "none", "pred": 0.5}))
        rows.append(pd.DataFrame({"Team": te.Team.values, "target_season": ty, "y": te.target.values,
                                  "model": "Benchmark: last year's win%", "features": "none",
                                  "pred": te.WinPct.values}))
        for fname, cols in feature_sets.items():
            for mname, m in models().items():
                if fname.startswith("A") and mname not in ("OLS",):
                    continue                                  # one feature: fancy models add nothing
                m.fit(tr[cols], tr.target)
                rows.append(pd.DataFrame({"Team": te.Team.values, "target_season": ty, "y": te.target.values,
                                          "model": mname, "features": fname, "pred": m.predict(te[cols])}))
    return pd.concat(rows, ignore_index=True)


def score(F):
    g = F.assign(err=F.pred - F.y).groupby(["features", "model"])
    out = g.agg(rmse=("err", lambda e: np.sqrt((e**2).mean())),
                mae=("err", lambda e: e.abs().mean()), n=("err", "size")).reset_index()
    base_mse = ((0.5 - F[F.model == "Benchmark: 0.500"].y) ** 2).mean()
    out["oos_r2_vs_0.500"] = 1 - out.rmse**2 / base_mse
    return out.sort_values("rmse").reset_index(drop=True)


def team_bootstrap_diff(F, m1, f1, m2, f2, B=2000, seed=SEED):
    """Paired team-cluster bootstrap of RMSE(model1) - RMSE(model2). Negative = model1 better."""
    a = F[(F.model == m1) & (F.features == f1)].set_index(["Team", "target_season"])
    b = F[(F.model == m2) & (F.features == f2)].set_index(["Team", "target_season"])
    e1, e2 = (a.pred - a.y) ** 2, (b.pred - b.y) ** 2
    e1, e2 = e1.align(e2)
    teams = np.array(sorted(set(e1.index.get_level_values(0))))
    rng = np.random.default_rng(seed)
    by_team1 = {t: e1.xs(t, level=0).to_numpy() for t in teams}
    by_team2 = {t: e2.xs(t, level=0).to_numpy() for t in teams}
    diffs = []
    for _ in range(B):
        s = rng.choice(teams, len(teams))
        m1_ = np.concatenate([by_team1[t] for t in s]).mean()
        m2_ = np.concatenate([by_team2[t] for t in s]).mean()
        diffs.append(np.sqrt(m1_) - np.sqrt(m2_))
    diffs = np.array(diffs)
    point = np.sqrt(e1.mean()) - np.sqrt(e2.mean())
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    d = load_pairs()
    fs = dict(FEATURE_SETS)
    fs["E all numeric"] = all_numeric(d)
    print(f"pairs: {len(d)}  | numeric features in E: {len(fs['E all numeric'])}")
    F = rolling_forecast(d, fs)
    F.to_csv(RES / "forecasts_out_of_sample.csv", index=False)
    S = score(F)
    S.to_csv(RES / "forecast_scores.csv", index=False)
    print(S.round(4).to_string(index=False))

    # paired comparisons with the persistence benchmark and the best model
    best = S[~S.model.str.startswith("Benchmark")].iloc[0]
    comps = []
    for (m, f) in [(best.model, best.features), ("Ridge", "D skill + payroll + win%"),
                   ("Ridge", "C skill metrics"), ("OLS", "B win% + payroll"),
                   ("GradBoost", "D skill + payroll + win%"), ("GradBoost", "E all numeric")]:
        for bm, bf in [("Benchmark: last year's win%", "none"), ("Benchmark: 0.500", "none"),
                       ("OLS", "A win% only")]:
            p, lo, hi = team_bootstrap_diff(F, m, f, bm, bf)
            comps.append({"model": m, "features": f, "versus": bm, "rmse_diff": p, "ci_lo": lo, "ci_hi": hi})
    C = pd.DataFrame(comps).drop_duplicates()
    C.to_csv(RES / "paired_rmse_differences.csv", index=False)
    print("\nPaired team-cluster bootstrap (RMSE difference, negative = model better):")
    print(C.round(4).to_string(index=False))

    # which inputs matter: standardized ridge coefficients on all data, permutation importance for GBM
    cols = fs["D skill + payroll + win%"]
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(d[cols], d.target)
    coef = pd.Series(ridge[-1].coef_, index=cols).sort_values()
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8,
                                   random_state=SEED).fit(d[cols], d.target)
    pi = permutation_importance(gb, d[cols], d.target, n_repeats=30, random_state=SEED)
    imp = pd.DataFrame({"ridge_std_coef": coef, "gb_perm_importance": pd.Series(pi.importances_mean, index=cols)})
    imp.to_csv(RES / "feature_importance_in_sample.csv")
    print("\nIn-sample importance (set D):\n", imp.round(4))

    # figures: full-axis bars (no truncated axis) plus paired differences with intervals
    key = S[(S.features == "none") | ((S.features == "A win% only")) |
            ((S.model.isin(["Ridge", "GradBoost"])) & (S.features.isin(["C skill metrics", "E all numeric"])))]
    key = key.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    lab = key.model + " | " + key.features
    ax[0].barh(lab, key.rmse, color=["#c00000" if m.startswith("Benchmark") else "#1f4e79" for m in key.model])
    ax[0].set_xlim(0, key.rmse.max() * 1.1)
    ax[0].set_xlabel("out-of-sample RMSE of next-season win% (lower is better)")
    ax[0].set_title("150 forecasts: 2019 and 2022-25")
    cc = C[C.versus == "OLS"].copy()
    cc["lab"] = cc.model + " | " + cc.features
    ax[1].errorbar(cc.rmse_diff, range(len(cc)), xerr=[cc.rmse_diff - cc.ci_lo, cc.ci_hi - cc.rmse_diff],
                   fmt="o", color="#1f4e79", capsize=3)
    ax[1].axvline(0, color="k", lw=.8); ax[1].set_yticks(range(len(cc))); ax[1].set_yticklabels(cc.lab, fontsize=8)
    ax[1].set_xlabel("RMSE difference vs shrunk win% (negative = better)")
    ax[1].set_title("Does anything beat shrunk win%? (95% team bootstrap)"); ax[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "forecast_rmse.png", dpi=160)

    r = F[(F.model == best.model) & (F.features == best.features)]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(r.pred, r.y, s=14, alpha=.6, color="#1f4e79")
    lim = [r[["pred", "y"]].min().min() - .01, r[["pred", "y"]].max().max() + .01]
    ax.plot(lim, lim, color="gray", lw=.8)
    ax.set_xlabel("forecast win%"); ax.set_ylabel("actual win%")
    ax.set_title(f"{best.model}, {best.features}")
    fig.tight_layout(); fig.savefig(FIG / "forecast_calibration.png", dpi=160)


if __name__ == "__main__":
    main()
