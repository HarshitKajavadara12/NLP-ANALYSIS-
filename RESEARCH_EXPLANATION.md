# Complete Research Report: Financial News NLP Analysis

## What This Project Is

A **research project that applies Natural Language Processing (NLP) to financial news articles** to extract insights, detect sentiment, identify events, and generate stock trading signals. The goal is to replicate what institutional hedge funds do using expensive proprietary platforms — but using free, open-source tools.

---

## Researcher's Thinking & Motivation

### Why This Was Started:
1. Hedge funds pay heavily for NLP platforms (RavenPack, Symphony, Bloomberg Terminal)
2. The techniques behind them are proprietary and secret
3. This project asks: Can open-source tools achieve similar outcomes?
4. Goal: Democratize financial NLP for anyone

### Core Hypothesis:
> "Can open-source NLP models (FinBERT, spaCy, VADER, LDA, XGBoost) produce useful signals from financial news analysis?"

---

## Actual Data Used (Truth Only)

| Metric | Actual Value |
|--------|-------------|
| Total Articles Collected | **63** |
| Unique Companies (Tickers) | **19** |
| Time Span | **114 days** |
| Average Article Length | **116 characters** |
| Average Word Count | **15 words** (these are headlines, not full articles) |
| Data Source | **Google News RSS only** |
| Data Completeness | **100%** (no missing titles, text, tickers, dates) |
| Average Articles Per Day | **3.15** |
| Max Articles in a Day | **12** |

### Companies Covered (19 tickers):
ABT (7 articles), BABA (6), AVGO (6), BAC (5), AMD (4), APA (4), BA (4), ACN (3), AMT (3), AVB (3), AZN (3), AAPL (2), ADBE (2), AMGN (2), AMZN (2), ASML (2), AXP (2), BBD (2), AEP (1)

### Data Quality:
- All 63 records are **headline-only** text (not full articles)
- Single source (Google News) introduces source bias
- Companies mostly start with 'A' — alphabetical collection bias from S&P 500 list
- Very short text (avg 15 words) limits what NLP can extract

---

## 9-Step Pipeline (What Was Actually Done)

### Step 1: Data Collection & Preprocessing
**Script:** `07_SCRIPTS/01_data_collection_and_preprocessing.py`

- Loads S&P 500 company list from Wikipedia
- Fetches company metadata via `yfinance`
- Scrapes news from Google News RSS feeds using `feedparser`
- Cleans text, removes duplicates, filters by quality (min 100 chars)
- Extracts temporal features (hour, day of week, market hours)

**Output:** 63 news articles with metadata in CSV format

---

### Step 2: Exploratory Data Analysis (EDA)
**Script:** `07_SCRIPTS/02_exploratory_data_analysis.py`

- Statistical analysis: article distribution by company, time, source
- Temporal patterns: busiest day = Friday, peak hour = 8 AM
- Coverage analysis: ABT most covered (7 articles), AEP least (1)
- Data quality assessment

**Finding:** Data is heavily skewed toward companies starting with 'A' (alphabetical collection order).

---

### Step 3: Sentiment Analysis (3 Models Compared)
**Script:** `07_SCRIPTS/03_sentiment_analysis_comprehensive.py`

**3 models used, each with different strengths:**

| Model | Type | Why Chosen |
|-------|------|-----------|
| **FinBERT** | Transformer (Deep Learning) | State-of-the-art for financial text; understands finance-specific language |
| **VADER** | Rule-based (Lexicon) | Fast, no training needed; good baseline |
| **TextBlob** | Pattern-based | Simple, general-purpose; comparison benchmark |

**Actual Sentiment Distribution:**

| Model | Positive % | Neutral % | Negative % | Avg Sentiment |
|-------|-----------|-----------|-----------|---------------|
| FinBERT | 49.2% | 28.6% | 22.2% | +0.115 |
| VADER | 49.2% | 28.6% | 22.2% | +0.115 |
| TextBlob | 31.7% | 63.5% | 4.8% | +0.094 |

**Note:** FinBERT and VADER produced identical distributions on this dataset. TextBlob classified most articles as neutral.

**Sentiment by Company (FinBERT):**
| Company | Avg Sentiment | Articles |
|---------|--------------|----------|
| BBD (Banco Bradesco) | +0.386 | 2 |
| BABA (Alibaba) | +0.330 | 6 |
| BAC (Bank of America) | +0.316 | 5 |
| AMD | +0.301 | 4 |
| ASML | +0.286 | 2 |
| BA (Boeing) | +0.252 | 4 |
| ADBE (Adobe) | -0.510 | 2 |
| AAPL (Apple) | -0.286 | 2 |

---

### Step 4: Named Entity Recognition (NER)
**Script:** `07_SCRIPTS/04_named_entity_recognition_comprehensive.py`

**2 NER models used in combination:**

| Model | Purpose |
|-------|---------|
| **spaCy** (`en_core_web_sm`) | Fast, general NER — companies, people, dates, money |
| **BERT NER** (`dslim/bert-base-NER`) | High precision for organizations, locations |

**Actual NER Results:**
| Metric | Value |
|--------|-------|
| Total Entities Extracted | **374** |
| Unique Entities | **226** |
| Avg Entities Per Article | **5.94** |
| Median Entities Per Article | **6** |

**Entity Types Found:**
| Entity Type | Count |
|-------------|-------|
| ORG (Organizations) | 156 |
| MONEY | 13 |
| DATE | 12 |
| MISC | 12 |
| PERSON | 10 |
| PERCENT | 6 |
| GPE (Locations) | 3 |
| LOC | 3 |
| PER | 3 |
| WORK_OF_ART | 3 |

**Top Entities Mentioned:**
| Entity | Count | Type |
|--------|-------|------|
| Yahoo Finance | 12 | ORG |
| Boeing | 7 | ORG |
| Abbott Laboratories | 7 | ORG |
| Bank of America | 6 | ORG |

**Co-mention Network:** Only 1 pair found: BA ↔ Boeing (10 co-mentions) — this is the same company mentioned two ways.

---

### Step 5: Event Detection & Classification
**Script:** `07_SCRIPTS/05_event_detection_classification_comprehensive.py`

**12 event categories defined** (M&A, Earnings, Product Launch, Regulatory, Executive Changes, Legal, Partnerships, Market Disruptions, Technology, ESG, Supply Chain, Competition)

**Method:** Keyword + regex pattern matching against `06_CONFIG/financial_events.json`

**Actual Results:**
| Metric | Value |
|--------|-------|
| Articles with events detected | **1 out of 63** |
| Coverage rate | **1.6%** |
| Event detected | EARNINGS_ANNOUNCEMENT |
| Impact distribution | 1 Medium, 62 Low |

**Why so low?** The articles are just headlines (15 words avg). Event detection rules need full-text articles with detailed language patterns to trigger.

---

### Step 6: Topic Modeling (LDA)
**Script:** `07_SCRIPTS/06_topic_modeling_classification_comprehensive.py`

**Configuration:**
- Algorithm: LDA (unsupervised)
- Number of topics: 10
- Passes: 10, Iterations: 400
- Vocabulary size: **3 words only** (price, finance, yahoo)

**Actual Model Metrics:**
| Metric | Value |
|--------|-------|
| Coherence Score | **0.636** |
| Perplexity | **-1.574** |
| Documents | 63 |
| Vocabulary | 3 words |

**Topic Distribution:**
- 54 articles → Topic 1 (finance, yahoo, price)
- 9 articles → Topic 4 (price, finance, yahoo)

**Reality:** All 10 topics contain the same 3 words (price, finance, yahoo) in different proportions. LDA failed to find meaningful topics because:
1. Only 3 unique meaningful words survived preprocessing
2. Headlines are too short for topic differentiation
3. All articles come from Google News (same formatting)

---

### Step 7: Advanced Feature Engineering (33 features created)
**Script:** `07_SCRIPTS/07_advanced_feature_engineering.py`

**93 total columns in final matrix (33 are engineered features):**

| Category | Count | Features |
|----------|-------|----------|
| Sentiment | 9 | sentiment_momentum, sentiment_ma_3d, sentiment_ma_7d, sentiment_ma_14d, sentiment_volatility, sentiment_range_7d, sentiment_acceleration, sentiment_extremity, sentiment_consensus |
| Entity | 5 | entity_density, entity_diversity, org_prominence, entity_freq_7d, entity_momentum |
| Event | 6 | event_density, has_high_impact_event, event_freq_7d, event_momentum, avg_event_impact_7d, event_sentiment_alignment |
| Topic | 4 | topic_entropy, topic_dominance, topic_shift, topic_stability |
| Temporal | 5 | hour, day_of_week, is_weekend, is_market_hours, days_since_last |
| Derived | 4 | recency_score, sentiment_lag_1, sentiment_lag_3, sentiment_lag_7 |

**Data Quality:**
- Missing values: 195 (completeness: 90.6%)
- Infinite values: 0

**Feature Importance (from XGBoost):**
| Feature | Importance |
|---------|-----------|
| sentiment_consensus | 22.2% |
| sentiment_extremity | 21.2% |
| sentiment_ma_7d | 13.0% |
| sentiment_momentum | 8.2% |
| entity_diversity | 7.2% |
| event_freq_7d | 5.9% |
| word_count | 3.7% |
| sentiment_volatility | 3.3% |
| is_market_hours | 3.1% |
| topic_entropy | 2.9% |

**Finding:** Sentiment features account for **68.6%** of total importance. Entity and event features contribute minimally.

---

### Step 8: Ensemble Modeling (8 base + 4 ensemble)
**Script:** `07_SCRIPTS/08_ensemble_modeling_strategies.py`

**Dataset Split:** 50 train / 13 test (80/20), 18 features used

#### Base Model Results:

| Model | Train Acc | Test Acc | Precision | F1-Score | Overfit Gap |
|-------|-----------|----------|-----------|----------|-------------|
| **XGBoost** | 100% | **92.3%** | 93.6% | 91.4% | 7.7% |
| Logistic Regression | 100% | 84.6% | 85.9% | 84.6% | 15.4% |
| Gradient Boosting | 100% | 84.6% | 84.6% | 84.6% | 15.4% |
| LightGBM | 100% | 84.6% | 84.6% | 84.6% | 15.4% |
| SVM | 98% | 84.6% | 89.0% | 84.3% | 13.4% |
| KNN | 100% | 84.6% | 89.0% | 84.3% | 15.4% |
| Random Forest | 100% | 76.9% | 71.8% | 74.1% | 23.1% |
| Naive Bayes | 56% | 53.8% | 50.0% | 48.1% | 2.2% |

#### Ensemble Results:

| Method | Test Accuracy | F1-Score |
|--------|--------------|----------|
| Hard Voting | 92.3% | 91.4% |
| Soft Voting | 92.3% | 91.4% |
| Stacking | 92.3% | 91.4% |
| Bagging | 76.9% | 74.1% |

**Key Observation:** All models except Naive Bayes have 100% train accuracy = severe overfitting on 50 samples. The 92.3% test accuracy on only 13 samples is not statistically reliable.

---

### Step 9: Trading Signal Generation
**Script:** `07_SCRIPTS/09_financial_news_nlp_pipeline.py`

**Signal Formula (weighted factors):**
| Factor | Weight |
|--------|--------|
| Sentiment (FinBERT + momentum + extremity) | 35% |
| Event Impact | 25% |
| News Velocity (article count + burst) | 20% |
| Entity Prominence (density + frequency) | 15% |
| Topic Relevance (dominance) | 5% |

**Signal Thresholds:**
| Score Range | Signal |
|-------------|--------|
| > 0.7 | Strong Buy |
| 0.4 to 0.7 | Buy |
| -0.4 to 0.4 | Neutral |
| -0.7 to -0.4 | Sell |
| < -0.7 | Strong Sell |

**Actual Signal Distribution:**
| Category | Count |
|----------|-------|
| Neutral | 42 (66.7%) |
| Strong Sell | 19 (30.2%) |
| Buy | 1 (1.6%) |
| Sell | 1 (1.6%) |
| Strong Buy | 0 (0%) |

**Trading Performance (actual):**
| Metric | Value |
|--------|-------|
| Win Rate | **50.8%** (target was 56.3%) |
| Max Drawdown | -13.2% |
| Sharpe Ratio | N/A (insufficient data) |
| Sortino Ratio | N/A |

**Most Bullish Companies:**
| Ticker | Avg Signal |
|--------|-----------|
| ASML | +0.307 |
| AXP | +0.209 |
| BBD | +0.180 |
| BABA | +0.155 |
| APA | +0.130 |

**Most Bearish Companies:**
| Ticker | Avg Signal |
|--------|-----------|
| ADBE | -0.269 |
| AMZN | -0.203 |
| AMT | -0.088 |
| ACN | -0.025 |

**Critical Issue:** 19 articles (30%) got "Strong Sell" signals, but many of these have POSITIVE FinBERT sentiment (e.g., Boeing +0.51, Bank of America +0.743). The signal calculation has a bug where missing feature values push the composite score below -0.7.

---

## Actual Results Summary (No False Claims)

| What Was Claimed (README) | What Was Actually Achieved |
|--------------------------|---------------------------|
| 500K+ articles | 63 articles |
| 12+ years of data | 114 days |
| 87.3% sentiment accuracy | Not validated (no ground truth) |
| 92.1% NER precision | Not formally evaluated |
| F1 = 0.823 event detection | 1.6% coverage, essentially failed |
| 88.9% ensemble accuracy | 92.3% on 13 test samples (unreliable) |
| 56.3% win rate | 50.8% (barely above random) |
| 150 articles/sec processing | Not benchmarked |
| Multiple data sources | Google News only |

---

## What Works, What Doesn't, What Needs Improvement

### What Actually Works:
- ✅ End-to-end pipeline architecture is sound
- ✅ Multi-model sentiment analysis produces reasonable results
- ✅ NER successfully extracts 374 entities from 63 articles
- ✅ Feature engineering creates meaningful derived features
- ✅ Code is modular and reproducible

### What Failed:
- ❌ Topic modeling — only 3 words in vocabulary, all topics identical
- ❌ Event detection — only 1 event found in 63 articles
- ❌ Trading signals — 30% incorrectly marked "Strong Sell"
- ❌ Win rate (50.8%) barely above random chance (50%)
- ❌ All model results are unreliable due to tiny dataset (13 test samples)

---

## Detailed System Improvement Recommendations

### 1. Data Scale (CRITICAL)
**Problem:** 63 articles is far too few for ML
**Solution:**
- Collect 10,000+ articles minimum
- Use multiple sources (NewsAPI, finnhub.io, Alpha Vantage, Reddit, Twitter/X)
- Collect full-text articles, not just headlines
- Target 500+ words per article for NLP to work properly
- Run collection over 6+ months for temporal patterns

### 2. Fix Data Collection Bias
**Problem:** Only companies starting with 'A' due to alphabetical S&P 500 scraping
**Solution:**
- Randomly sample 50-100 companies from all sectors
- Ensure sector balance (tech, healthcare, energy, finance, consumer)
- Add market cap diversity (mega, large, mid, small)

### 3. Full-Text Article Retrieval
**Problem:** Headlines average 15 words — too short for NLP
**Solution:**
- Use newspaper3k or trafilatura to extract full article text
- Target articles with 200+ words
- Store both headline and body separately
- This alone would fix topic modeling and event detection

### 4. Fix Topic Modeling
**Problem:** Vocabulary of only 3 words (price, finance, yahoo) — preprocessing removed everything meaningful
**Solution:**
- Fix text preprocessing: don't over-aggressively remove words
- Use article body text (not headlines)
- Keep financial terms, company names, industry jargon
- Reduce stop-word removal aggressiveness
- Try BERTopic instead of LDA (works better on short text)
- Reduce topics from 10 to 5 (for small datasets)

### 5. Fix Event Detection
**Problem:** 1.6% detection rate — regex patterns need full text to match
**Solution:**
- Collect full-text articles (patterns like "Q1 earnings beat" appear in body)
- Add machine learning classifier trained on labeled event examples
- Use FinBERT zero-shot classification for event types
- Expand keyword dictionaries with more variations

### 6. Fix Trading Signal Bug
**Problem:** 30% of articles get "Strong Sell" even with positive sentiment
**Solution:**
- Handle NaN/missing values before signal calculation (impute with 0 not NaN)
- Validate that each factor contributes correctly
- Add safeguard: if sentiment is positive, signal cannot be Strong Sell
- Test signal formula on known cases before production

### 7. Add Stock Price Data
**Problem:** No actual price correlation or backtesting
**Solution:**
- Fetch historical stock prices via `yfinance`
- Calculate next-day, next-3-day, next-week returns after each article
- Use actual returns as the ML target variable (not synthetic)
- Implement proper backtesting with train/test time-split

### 8. Fix Overfitting
**Problem:** All models achieve 100% train accuracy — severe memorization
**Solution:**
- Get more data (10,000+ samples minimum)
- Use proper time-series cross-validation (no future leak)
- Increase regularization (reduce max_depth, increase min_samples)
- Use feature selection to reduce 18 features to top 8-10
- Report validation curves, not just test accuracy

### 9. Improve Evaluation
**Problem:** 13 test samples is statistically meaningless
**Solution:**
- With more data, use 5-fold cross-validation
- Use time-series split (train on past, test on future)
- Report confidence intervals, not point estimates
- Compare against baselines (random, majority class, always-neutral)

### 10. Add Real-Time Pipeline
**Problem:** Batch processing only — no live signal generation
**Solution:**
- Add scheduler (cron/APScheduler) for hourly data collection
- Stream processing for incoming news
- Alert system for Strong Buy/Sell signals
- Dashboard for real-time monitoring

---

## Technologies & Libraries Used

### Core NLP:
- `transformers` + `ProsusAI/finbert` — Financial sentiment transformer
- `spacy` + `en_core_web_sm` — Named Entity Recognition
- `dslim/bert-base-NER` — BERT-based NER
- `vaderSentiment` — Rule-based sentiment
- `textblob` — Pattern-based sentiment
- `gensim` — LDA topic modeling

### Machine Learning:
- `scikit-learn` — Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, ensembles
- `xgboost` — XGBoost classifier
- `lightgbm` — LightGBM classifier

### Data Collection:
- `feedparser` — Google News RSS parsing
- `yfinance` — Company metadata
- `pandas`, `numpy` — Data manipulation

### Visualization:
- `matplotlib`, `seaborn`, `plotly` — Charts and graphs

---

## Project Structure

```
NLP-ANALYSIS-/
│
├── 01_DATA/
│   ├── raw/                    # Original 63 articles CSV
│   ├── processed/              # (empty — inline processing)
│   └── features/               # Enriched datasets at each pipeline stage
│       ├── dataset_with_sentiment.csv
│       ├── dataset_with_entities.csv
│       ├── dataset_with_events.csv
│       ├── dataset_with_topics.csv
│       └── final_feature_matrix.csv   (93 columns)
│
├── 02_NOTEBOOKS/               # 9 Jupyter notebooks (interactive versions)
├── 03_RESULTS/
│   ├── metrics/                # JSON: collection stats, EDA summary
│   ├── outputs/                # CSV/JSON: all model results, signals
│   └── visualizations/         # 32 HTML/PNG charts
│
├── 04_MODELS/                  # Trained LDA model (gensim format)
├── 06_CONFIG/                  # All configuration (framework, ensemble, events, lexicon)
└── 07_SCRIPTS/                 # 9 standalone Python scripts
```

---

## How to Reproduce

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cd 07_SCRIPTS
python 01_data_collection_and_preprocessing.py
python 02_exploratory_data_analysis.py
python 03_sentiment_analysis_comprehensive.py
python 04_named_entity_recognition_comprehensive.py
python 05_event_detection_classification_comprehensive.py
python 06_topic_modeling_classification_comprehensive.py
python 07_advanced_feature_engineering.py
python 08_ensemble_modeling_strategies.py
python 09_financial_news_nlp_pipeline.py
```

---

## Final Conclusion

This is a **proof-of-concept project** that demonstrates the architecture hedge funds use for news-based trading signals. The code infrastructure is well-organized and modular, but the execution was limited by:

1. **Too little data** (63 articles instead of thousands)
2. **Headlines only** (15 words avg, not full articles)
3. **Single source** (Google News only)
4. **No price validation** (no actual stock price backtesting)

The 92.3% model accuracy is unreliable (only 13 test samples, overfitting). The real win rate of 50.8% shows the signals barely beat random chance. However, the **architecture is correct** — with 10,000+ full-text articles and proper price data, this pipeline could produce genuinely useful results.
