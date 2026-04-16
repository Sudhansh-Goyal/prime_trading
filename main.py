"""
FastAPI — minimal REST layer over the analysis module.
All endpoints return JSON; no heavy processing happens here.
"""
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import app.analysis as a
from app.report import build_report

app = FastAPI(
    title="Bitcoin Sentiment × Trader Performance API",
    description="Hyperliquid trade data analysed against the Fear & Greed Index.",
    version="2.0.0",
)


@app.get("/", response_class=HTMLResponse)
def report():
    """Full visual analysis report with all 9 Plotly charts."""
    return build_report()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """High-level dataset KPIs."""
    return a.key_stats()


@app.get("/sentiment-pnl")
def sentiment_pnl():
    """Descriptive PnL stats (mean, median, std, win-rate, profit-factor) per sentiment."""
    return a.sentiment_pnl_stats()


@app.get("/anova")
def anova():
    """ANOVA + pairwise Bonferroni t-tests across sentiment groups."""
    return a.anova_across_sentiments()


@app.get("/contrarian")
def contrarian():
    """Contrarian (buy-fear/sell-greed) vs Momentum strategy PnL comparison."""
    return a.contrarian_analysis()


@app.get("/clusters")
def clusters(n: int = Query(default=4, ge=2, le=8)):
    """KMeans trader-behaviour clustering (n clusters)."""
    return a.trader_clusters(n_clusters=n)


@app.get("/classifier")
def classifier():
    """RandomForest trade-profitability classifier — CV accuracy & feature importances."""
    return a.profitability_classifier()


@app.get("/rolling-correlation")
def rolling_corr(window: int = Query(default=30, ge=7, le=90)):
    """Pearson correlation + rolling series between sentiment score and daily PnL."""
    return a.rolling_correlation(window=window)


@app.get("/position-sizing")
def position_sizing():
    """Mean/median/p90 trade size and long-short ratio per sentiment class."""
    return a.position_sizing_by_sentiment()


@app.get("/top-traders-by-regime")
def top_traders_regime(n: int = Query(default=5, ge=1, le=20)):
    """Top-N traders per sentiment regime."""
    return a.top_traders_by_regime(n=n)
