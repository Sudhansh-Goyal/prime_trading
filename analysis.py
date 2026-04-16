"""
Core analysis module
====================
Joins Hyperliquid trader data with Bitcoin Fear & Greed Index then runs:

  1.  Descriptive statistics per sentiment regime
  2.  ANOVA + pairwise Tukey HSD (are sentiment groups statistically different?)
  3.  Contrarian-signal study (trade direction vs sentiment)
  4.  Trader-behaviour clustering (KMeans)
  5.  Trade-profitability classifier (RandomForest feature importance)
  6.  Pearson correlation + rolling 30-day correlation (sentiment score vs PnL)
  7.  Position-sizing behaviour by sentiment
  8.  Regime-aware top-trader leaderboard
"""

import os, warnings
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Any, Dict, List

from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════════

def _load_fear_greed() -> pd.DataFrame:
    path = os.path.join(DATASETS_DIR, "fear_greed_index.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["classification"] = df["classification"].str.strip().str.title()
    return df[["date", "value", "classification"]].rename(
        columns={"value": "sentiment_score", "classification": "sentiment"}
    )


def _load_trades() -> pd.DataFrame:
    path = os.path.join(DATASETS_DIR, "historical_data.csv")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Account": "account", "Coin": "coin",
        "Execution Price": "exec_price", "Size Tokens": "size_tokens",
        "Size USD": "size_usd", "Side": "side",
        "Timestamp IST": "ts_ist", "Start Position": "start_position",
        "Direction": "direction", "Closed PnL": "closed_pnl", "Fee": "fee",
    })
    df["trade_date"] = pd.to_datetime(
        df["ts_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
    ).dt.date
    for col in ["exec_price", "size_tokens", "size_usd", "closed_pnl", "fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["side"] = df["side"].str.strip().str.upper()
    df["direction"] = df["direction"].str.strip().str.title()
    return df


@lru_cache(maxsize=1)
def merged() -> pd.DataFrame:
    """Full dataset: every trade row merged with its-day sentiment score."""
    fg = _load_fear_greed()
    tr = _load_trades()
    return tr.merge(fg, left_on="trade_date", right_on="date", how="left")


@lru_cache(maxsize=1)
def closed_trades() -> pd.DataFrame:
    """Only rows where a position was actually closed (PnL ≠ 0)."""
    df = merged()
    return df[df["closed_pnl"] != 0].copy()


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DESCRIPTIVE STATS BY SENTIMENT REGIME
# ═══════════════════════════════════════════════════════════════════════════

def sentiment_pnl_stats() -> List[Dict]:
    """
    Per-sentiment descriptive statistics:
    mean, median, std, win-rate, total PnL, trade count.
    """
    df = closed_trades()
    df = df[df["sentiment"].notna()]
    df["win"] = (df["closed_pnl"] > 0).astype(int)

    result = []
    for sentiment, grp in df.groupby("sentiment"):
        pnl = grp["closed_pnl"]
        result.append({
            "sentiment": sentiment,
            "trade_count": int(len(pnl)),
            "total_pnl": round(float(pnl.sum()), 2),
            "mean_pnl": round(float(pnl.mean()), 4),
            "median_pnl": round(float(pnl.median()), 4),
            "std_pnl": round(float(pnl.std()), 4),
            "win_rate_pct": round(float(grp["win"].mean() * 100), 2),
            "profit_factor": round(
                float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()))
                if pnl[pnl < 0].sum() != 0 else float("inf"), 4
            ),
        })
    return sorted(result, key=lambda x: x["total_pnl"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  STATISTICAL SIGNIFICANCE: ANOVA + PAIRWISE T-TESTS
# ═══════════════════════════════════════════════════════════════════════════

def anova_across_sentiments() -> Dict:
    """
    One-way ANOVA: tests whether mean PnL differs across sentiment classes.
    Also runs pairwise t-tests (with Bonferroni correction) between every
    pair of sentiment groups.
    """
    df = closed_trades()
    df = df[df["sentiment"].notna()]
    groups = {s: g["closed_pnl"].values for s, g in df.groupby("sentiment")}

    f_stat, p_value = f_oneway(*groups.values())

    sentiments = list(groups.keys())
    pairwise = []
    n_pairs = len(sentiments) * (len(sentiments) - 1) // 2
    for i in range(len(sentiments)):
        for j in range(i + 1, len(sentiments)):
            a, b = sentiments[i], sentiments[j]
            t, p = ttest_ind(groups[a], groups[b], equal_var=False)
            p_bonf = min(p * n_pairs, 1.0)          # Bonferroni corrected
            pairwise.append({
                "group_a": a, "group_b": b,
                "t_statistic": round(float(t), 4),
                "p_value_raw": round(float(p), 6),
                "p_value_bonferroni": round(float(p_bonf), 6),
                "significant_at_5pct": bool(p_bonf < 0.05),
            })

    return {
        "anova": {
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_value), 8),
            "significant_at_5pct": bool(p_value < 0.05),
            "interpretation": (
                "Sentiment class significantly affects mean PnL per trade."
                if p_value < 0.05
                else "No statistically significant difference in mean PnL across sentiments."
            ),
        },
        "pairwise_t_tests": sorted(pairwise, key=lambda x: x["p_value_bonferroni"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  CONTRARIAN SIGNAL STUDY
# ═══════════════════════════════════════════════════════════════════════════

def contrarian_analysis() -> Dict:
    """
    Tests the "buy fear, sell greed" hypothesis.

    Contrarian = buy (long) during Fear/Extreme Fear
                 sell (short) during Greed/Extreme Greed.
    Momentum   = the opposite direction.

    Returns average PnL for contrarian vs momentum trades,
    plus win-rate comparison.
    """
    df = closed_trades()
    df = df[df["sentiment"].notna() & df["side"].notna()]

    fear_mask = df["sentiment"].isin(["Fear", "Extreme Fear"])
    greed_mask = df["sentiment"].isin(["Greed", "Extreme Greed"])

    # Contrarian: long in fear, short in greed
    contrarian = df[
        (fear_mask & (df["side"] == "BUY")) |
        (greed_mask & (df["side"] == "SELL"))
    ]
    # Momentum: long in greed, short in fear
    momentum = df[
        (greed_mask & (df["side"] == "BUY")) |
        (fear_mask & (df["side"] == "SELL"))
    ]
    neutral = df[df["sentiment"] == "Neutral"]

    def stats(grp: pd.DataFrame, label: str) -> Dict:
        pnl = grp["closed_pnl"]
        return {
            "strategy": label,
            "trade_count": int(len(pnl)),
            "total_pnl": round(float(pnl.sum()), 2),
            "mean_pnl": round(float(pnl.mean()), 4),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "std_pnl": round(float(pnl.std()), 4),
        }

    t_stat, p_val = ttest_ind(
        contrarian["closed_pnl"].values,
        momentum["closed_pnl"].values,
        equal_var=False,
    )

    return {
        "results": [
            stats(contrarian, "Contrarian (buy-fear / sell-greed)"),
            stats(momentum,   "Momentum  (buy-greed / sell-fear)"),
            stats(neutral,    "Neutral-day trades"),
        ],
        "t_test_contrarian_vs_momentum": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_at_5pct": bool(p_val < 0.05),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  TRADER BEHAVIOURAL CLUSTERING (KMeans)
# ═══════════════════════════════════════════════════════════════════════════

def trader_clusters(n_clusters: int = 4) -> Dict:
    """
    Builds a per-trader feature matrix:
      - total_pnl, win_rate, avg_trade_size, trade_count,
        avg_sentiment_score (weighted by trade), long_ratio
    Then runs KMeans and labels each cluster.
    """
    df = closed_trades()
    df = df[df["sentiment"].notna()]
    df["win"] = (df["closed_pnl"] > 0).astype(int)
    df["is_long"] = (df["side"] == "BUY").astype(int)

    agg = df.groupby("account").agg(
        total_pnl=("closed_pnl", "sum"),
        win_rate=("win", "mean"),
        avg_trade_size=("size_usd", "mean"),
        trade_count=("closed_pnl", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        long_ratio=("is_long", "mean"),
    ).reset_index()

    # Only cluster traders with enough trades
    agg = agg[agg["trade_count"] >= 10].copy()

    features = ["total_pnl", "win_rate", "avg_trade_size",
                "trade_count", "avg_sentiment", "long_ratio"]
    X = agg[features].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    agg["cluster"] = km.fit_predict(X_scaled)

    # Summarise each cluster
    cluster_summary = []
    for cid, grp in agg.groupby("cluster"):
        cluster_summary.append({
            "cluster_id": int(cid),
            "trader_count": int(len(grp)),
            "avg_total_pnl": round(float(grp["total_pnl"].mean()), 2),
            "avg_win_rate_pct": round(float(grp["win_rate"].mean() * 100), 2),
            "avg_trade_size_usd": round(float(grp["avg_trade_size"].mean()), 2),
            "avg_trades_per_trader": round(float(grp["trade_count"].mean()), 1),
            "avg_sentiment_score": round(float(grp["avg_sentiment"].mean()), 2),
            "avg_long_ratio_pct": round(float(grp["long_ratio"].mean() * 100), 2),
            "profile": _cluster_profile(grp),
        })

    # Per-trader cluster assignment
    trader_labels = agg[["account", "cluster", "total_pnl",
                          "win_rate", "trade_count"]].copy()
    trader_labels["win_rate"] = (trader_labels["win_rate"] * 100).round(2)
    trader_labels["total_pnl"] = trader_labels["total_pnl"].round(2)

    return {
        "n_clusters": n_clusters,
        "inertia": round(float(km.inertia_), 2),
        "cluster_summary": sorted(cluster_summary,
                                  key=lambda x: x["avg_total_pnl"], reverse=True),
        "trader_assignments": trader_labels.to_dict(orient="records"),
    }


def _cluster_profile(grp: pd.DataFrame) -> str:
    pnl = grp["total_pnl"].mean()
    wr  = grp["win_rate"].mean()
    sz  = grp["avg_trade_size"].mean()
    lr  = grp["long_ratio"].mean()

    if pnl > 0 and wr > 0.80:
        label = "Elite Profitable Traders"
    elif pnl > 0 and sz > grp["avg_trade_size"].median() * 2:
        label = "High-Volume Winners"
    elif pnl < 0 and wr < 0.60:
        label = "Struggling / High-Risk Traders"
    elif lr > 0.70:
        label = "Long-Biased Traders"
    elif lr < 0.30:
        label = "Short-Biased Traders"
    else:
        label = "Balanced / Neutral Traders"
    return label


# ═══════════════════════════════════════════════════════════════════════════
# 6.  TRADE-PROFITABILITY CLASSIFIER (RandomForest)
# ═══════════════════════════════════════════════════════════════════════════

def profitability_classifier() -> Dict:
    """
    Trains a RandomForestClassifier to predict whether a trade will be
    profitable (closed_pnl > 0) using:
      sentiment_score, size_usd, side (encoded), exec_price
    Returns 5-fold CV accuracy and feature importances.
    """
    df = closed_trades()
    df = df[df["sentiment"].notna()].copy()

    df["is_buy"] = (df["side"] == "BUY").astype(int)
    df["target"] = (df["closed_pnl"] > 0).astype(int)

    features = ["sentiment_score", "size_usd", "exec_price", "is_buy"]
    df_model = df[features + ["target"]].dropna()

    X = df_model[features].values
    y = df_model["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")

    # Fit on full data for feature importance
    clf.fit(X_scaled, y)
    importances = sorted(
        zip(features, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )

    # Class distribution
    class_dist = {
        "profitable_trades": int(y.sum()),
        "loss_trades": int((y == 0).sum()),
        "class_balance_ratio": round(float(y.sum() / len(y)), 4),
    }

    return {
        "model": "RandomForestClassifier (200 trees, max_depth=6)",
        "feature_set": features,
        "cross_val_accuracy_scores": [round(s, 4) for s in cv_scores],
        "mean_cv_accuracy": round(float(cv_scores.mean()), 4),
        "std_cv_accuracy": round(float(cv_scores.std()), 4),
        "feature_importances": [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in importances
        ],
        "class_distribution": class_dist,
        "interpretation": (
            f"The model predicts trade profitability with "
            f"{round(cv_scores.mean()*100, 1)}% accuracy (5-fold CV). "
            f"'{importances[0][0]}' is the most predictive feature."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7.  ROLLING CORRELATION (sentiment score vs daily PnL)
# ═══════════════════════════════════════════════════════════════════════════

def rolling_correlation(window: int = 30) -> Dict:
    """
    Computes overall Pearson correlation AND a 30-day rolling correlation
    between the daily sentiment score and the daily aggregate PnL.
    Returns the rolling series (useful for charting) and the overall figure.
    """
    df = closed_trades()
    df = df[df["sentiment_score"].notna()]

    daily = (
        df.groupby("trade_date")
        .agg(daily_pnl=("closed_pnl", "sum"),
             sentiment_score=("sentiment_score", "first"))
        .reset_index()
        .sort_values("trade_date")
    )

    overall_corr = float(
        daily[["sentiment_score", "daily_pnl"]].corr().iloc[0, 1]
    )

    daily["rolling_corr"] = (
        daily["sentiment_score"]
        .rolling(window)
        .corr(daily["daily_pnl"])
    )

    rolling_records = (
        daily[["trade_date", "rolling_corr", "daily_pnl", "sentiment_score"]]
        .dropna()
        .tail(180)           # last 180 days for presentation
        .copy()
    )
    rolling_records["trade_date"] = rolling_records["trade_date"].astype(str)
    rolling_records = rolling_records.round(4)

    return {
        "overall_pearson_r": round(overall_corr, 4),
        "interpretation": (
            "Traders profit more when sentiment is fearful (contrarian edge)."
            if overall_corr < -0.05
            else "Traders profit more when market is greedy (momentum edge)."
            if overall_corr > 0.05
            else "Sentiment score shows negligible linear relationship with daily PnL."
        ),
        "rolling_window_days": window,
        "rolling_series": rolling_records.to_dict(orient="records"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8.  POSITION-SIZING BEHAVIOUR BY SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════

def position_sizing_by_sentiment() -> List[Dict]:
    """
    Do traders size positions differently under fear vs greed?
    Returns mean / median / 90th-pct trade size per sentiment class.
    """
    df = merged()
    df = df[df["sentiment"].notna() & df["size_usd"].notna()]

    result = []
    for sentiment, grp in df.groupby("sentiment"):
        sz = grp["size_usd"]
        result.append({
            "sentiment": sentiment,
            "trade_count": int(len(sz)),
            "mean_size_usd": round(float(sz.mean()), 2),
            "median_size_usd": round(float(sz.median()), 2),
            "p90_size_usd": round(float(np.percentile(sz.dropna(), 90)), 2),
            "p99_size_usd": round(float(np.percentile(sz.dropna(), 99)), 2),
            "long_pct": round(
                float((grp["side"] == "BUY").mean() * 100), 2
            ),
            "short_pct": round(
                float((grp["side"] == "SELL").mean() * 100), 2
            ),
        })
    return sorted(result, key=lambda x: x["mean_size_usd"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  REGIME-AWARE TOP TRADERS
# ═══════════════════════════════════════════════════════════════════════════

def top_traders_by_regime(n: int = 5) -> Dict:
    """
    For each sentiment regime, who are the top-N traders by total PnL?
    Useful for identifying specialists (e.g. traders who only shine in fear).
    """
    df = closed_trades()
    df = df[df["sentiment"].notna()]

    result = {}
    for sentiment, sdf in df.groupby("sentiment"):
        grp = (
            sdf.groupby("account")
            .agg(total_pnl=("closed_pnl", "sum"),
                 trade_count=("closed_pnl", "count"),
                 win_rate=("closed_pnl", lambda x: round((x > 0).mean() * 100, 2)))
            .reset_index()
            .sort_values("total_pnl", ascending=False)
            .head(n)
        )
        grp["total_pnl"] = grp["total_pnl"].round(2)
        result[sentiment] = grp.to_dict(orient="records")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 10.  KEY STATS SUMMARY (for the overview endpoint)
# ═══════════════════════════════════════════════════════════════════════════

def key_stats() -> Dict:
    df = merged()
    cd = closed_trades()
    fg = _load_fear_greed()

    daily_pnl = (
        cd.groupby(["trade_date", "sentiment", "sentiment_score"])["closed_pnl"]
        .sum()
        .reset_index()
        .sort_values("closed_pnl", ascending=False)
    )

    return {
        "total_trades": int(len(df)),
        "closed_trades": int(len(cd)),
        "unique_traders": int(df["account"].nunique()),
        "unique_coins": int(df["coin"].nunique()),
        "total_pnl_usd": round(float(cd["closed_pnl"].sum()), 2),
        "overall_win_rate_pct": round(float((cd["closed_pnl"] > 0).mean() * 100), 2),
        "total_volume_usd": round(float(df["size_usd"].sum()), 2),
        "sentiment_date_range": {
            "from": str(fg["date"].min()),
            "to": str(fg["date"].max()),
        },
        "best_day": {
            "date": str(daily_pnl.iloc[0]["trade_date"]),
            "sentiment": daily_pnl.iloc[0]["sentiment"],
            "pnl_usd": round(float(daily_pnl.iloc[0]["closed_pnl"]), 2),
        },
        "worst_day": {
            "date": str(daily_pnl.iloc[-1]["trade_date"]),
            "sentiment": daily_pnl.iloc[-1]["sentiment"],
            "pnl_usd": round(float(daily_pnl.iloc[-1]["closed_pnl"]), 2),
        },
    }
