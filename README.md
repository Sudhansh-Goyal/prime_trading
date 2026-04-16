# Bitcoin Sentiment × Hyperliquid Trader Analysis

Explores the relationship between the **Bitcoin Fear & Greed Index** and real trader performance data from **Hyperliquid**, using statistical testing, ML models, and an interactive visual report served via FastAPI.

---

## Datasets

| File | Description |
|------|-------------|
| `datasets/fear_greed_index.csv` | Daily Bitcoin Fear & Greed Index — columns: `timestamp`, `value`, `classification`, `date` |
| `datasets/historical_data.csv` | ~211K Hyperliquid trades — columns: `Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Timestamp IST`, `Closed PnL`, `Fee`, etc. |

---

## Project Structure

```
prime-trading-ds-intern/
├── datasets/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
├── app/
│   ├── __init__.py
│   ├── analysis.py      # All ML & statistical analysis logic
│   ├── report.py        # Plotly chart builders → HTML report
│   └── main.py          # FastAPI routes (thin layer)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Analysis Modules (`app/analysis.py`)

### 1. Descriptive Statistics by Sentiment Regime
Per sentiment class (Extreme Fear → Extreme Greed): mean PnL, median PnL, std, win-rate, and profit factor.

### 2. ANOVA + Pairwise t-Tests
One-way ANOVA tests whether mean PnL differs significantly across sentiment classes.  
Bonferroni-corrected pairwise t-tests identify which pairs are statistically different.  
**Result:** F=7.74, p=3.14×10⁻⁶ — sentiment class significantly affects per-trade PnL.

### 3. Contrarian Signal Study
Tests the "buy fear, sell greed" hypothesis head-to-head against momentum trading.

| Strategy | Mean PnL | Win Rate | t-stat |
|----------|----------|----------|--------|
| Contrarian (buy-fear / sell-greed) | **$157.15** | **85.6%** | 10.44 |
| Momentum (buy-greed / sell-fear) | $56.69 | 81.4% | — |

p-value ≈ 0 → contrarian edge is statistically significant.

### 4. KMeans Trader Behaviour Clustering
Clusters all traders (min 10 trades) on 6 features: total PnL, win-rate, avg trade size, trade count, avg sentiment exposure, long/short ratio.  
Returns cluster profiles and per-trader assignments.

### 5. RandomForest Profitability Classifier
Predicts whether a trade will be profitable using: `sentiment_score`, `size_usd`, `exec_price`, `is_buy`.

| Metric | Value |
|--------|-------|
| Mean CV Accuracy (5-fold) | 58.97% |
| Top Feature | `exec_price` (46.5%) |
| 2nd Feature | `sentiment_score` (41.5%) |

### 6. Rolling Correlation
30-day rolling Pearson correlation between daily sentiment score and daily aggregate PnL.  
Overall r = **−0.098** → traders net slightly more profit on fearful days (contrarian edge confirmed).

### 7. Position Sizing by Sentiment
Traders size positions **~2.5× larger during Fear** ($7,816 mean) vs Extreme Greed ($3,112 mean).

### 8. Regime-Aware Top Trader Leaderboard
Top-N traders ranked by total PnL, broken down per sentiment regime — identifies sentiment specialists.

---

## Key Findings

- **ANOVA is significant** (F=7.74, p=3.14×10⁻⁶): sentiment class statistically affects per-trade PnL
- **Contrarian edge confirmed**: buy-fear/sell-greed yields 2.8× higher mean PnL than momentum
- **Fear** days produce the highest total PnL ($3.36M); **Extreme Greed** has the highest win-rate (89.2%)
- **Sentiment score** is the 2nd most predictive feature for trade profitability (42% RF importance)
- **Traders consciously size up in fear**: mean trade size in Fear is 36% above the dataset average
- **Correlation = −0.098**: slight but consistent contrarian advantage across all trading days

---

## Visual Report

The root endpoint `/` serves a self-contained interactive HTML report with **9 Plotly charts**:

1. Total PnL & Win Rate by Sentiment (bar charts)
2. Mean PnL per Trade with SEM error bars + median overlay
3. Profit Factor by Sentiment
4. Contrarian vs Momentum strategy comparison
5. Pairwise t-test p-value heatmap (Bonferroni corrected)
6. 30-day Rolling Correlation: sentiment score vs daily PnL
7. Position sizing & long/short ratio by sentiment
8. RandomForest feature importances + 5-fold CV accuracy
9. KMeans trader behaviour clustering (bubble chart)

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive HTML report with all 9 charts |
| GET | `/health` | Health check |
| GET | `/stats` | Dataset KPIs |
| GET | `/sentiment-pnl` | Descriptive stats per sentiment class |
| GET | `/anova` | ANOVA + pairwise Bonferroni t-tests |
| GET | `/contrarian` | Contrarian vs Momentum strategy results |
| GET | `/clusters?n=4` | KMeans trader clustering (n=2–8) |
| GET | `/classifier` | RF model CV accuracy & feature importances |
| GET | `/rolling-correlation?window=30` | Rolling Pearson r series |
| GET | `/position-sizing` | Trade size distribution by sentiment |
| GET | `/top-traders-by-regime?n=5` | Top-N traders per sentiment regime |
| GET | `/docs` | Swagger UI |

---

## Running Locally

**Prerequisites:** Python 3.11+

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Open report
open http://localhost:8001
```

---

## Running with Docker

```bash
# Build and start
docker compose up --build

# Report available at
open http://localhost:8001
```

Or manually:

```bash
docker build -t prime-trading-api .
docker run -p 8001:8000 prime-trading-api
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, merging, groupby analysis |
| `numpy` | Numerical operations |
| `scipy` | ANOVA, pairwise t-tests, Bonferroni correction |
| `scikit-learn` | KMeans clustering, RandomForest classifier, cross-validation |
| `plotly` | Interactive chart generation |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
