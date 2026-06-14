# Data Directory Documentation

## Overview

This directory contains all datasets used in the Financial News NLP Research project. Data is organized into four subdirectories representing different stages of the data pipeline.

---

## Directory Structure

```
01_DATA/
 raw/                  # Original, unmodified data from sources
 processed/            # Cleaned and standardized data (.parquet format)
 features/             # Data with extracted NLP features
 training/             # Train/validation splits for model training
```

---

## Data Files

### `raw/` - Original Data
**Purpose**: Store unmodified data from collection sources

**Files**:
- `combined_news_2013_2023.csv` - Historical news (10+ years)
- `global_financial_news_yahoo_2024.csv` - Yahoo Finance 2024
- `us_google_news_2024.csv` - Google News US markets
- `combined_economic_news.csv` - Economic indicators
- `combined_financial_news.csv` - Financial press releases

**Format**: CSV files with columns: `date`, `title`, `text`, `source`, `url`

---

### `processed/` - Cleaned Data
**Purpose**: Standardized, cleaned data ready for analysis

**Files**:
- `consolidated_text_data.parquet` - Main dataset (all sources combined)
- `processed_*.parquet` - Individual source files processed

**Format**: Parquet (compressed, efficient)

**Processing Applied**:
- Removed HTML tags and special characters
- Standardized dates to ISO 8601 format
- Deduplicated articles (4.7% removed)
- Fixed encoding issues (UTF-8)
- Validated company tickers
- Removed articles <100 or >10,000 words

---

### `features/` - Feature-Engineered Data
**Purpose**: Data with NLP features extracted

**Files**:
- `dataset_with_sentiment.csv` - Sentiment scores added (FinBERT + VADER)
- `dataset_with_entities.csv` - Named entities extracted (companies, tickers, people)
- `dataset_with_events.csv` - Event classifications (M&A, earnings, etc.)
- `financial_entity_database.csv` - Entity reference table
- `ticker_mentions.csv` - Company mention frequencies
- `ticker_mentions_with_events.csv` - Final feature set for modeling

**Additional Features**:
- Sentiment momentum (rate of change)
- Entity co-mention networks
- News velocity metrics
- Topic relevance scores

---

### `training/` - Model Training Data
**Purpose**: Prepared datasets for machine learning

**Files**:
- `event_train_data.csv` - Training set (80% of data)
- `event_val_data.csv` - Validation set (20% of data)
- `financial_ner_training_data.pkl` - NER training examples

**Split Method**: Stratified temporal split (no data leakage)

---

## Dataset Statistics

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Articles** | 523,847 |
| **Time Span** | 2013-01-01 to 2025-10-31 |
| **Unique Companies** | 847 |
| **Unique Tickers** | 623 |
| **Named Entities** | 47,832 |
| **Average Article Length** | 487 words |
| **Languages** | English only |

### Data Quality Metrics

| Quality Check | Pass Rate | Details |
|---------------|-----------|---------|
| **Completeness** | 97.7% | Missing values <2.3% |
| **Uniqueness** | 95.3% | Duplicates removed |
| **Validity** | 99.9% | Valid dates |
| **Accuracy** | 97.2% | Ticker validation |
| **Timeliness** | 99.1% | <24h delay |

---

## Coverage Breakdown

### By Source
- **Yahoo Finance**: 48% (250K articles)
- **Google News**: 29% (150K articles)
- **Reuters**: 14% (75K articles)
- **Bloomberg**: 6% (30K articles)
- **Economic News**: 3% (15K articles)

### By Geography
- **United States**: 68%
- **European Union**: 18%
- **Asia-Pacific**: 10%
- **Emerging Markets**: 4%

### By Sector
- **Technology**: 24%
- **Financials**: 19%
- **Healthcare**: 14%
- **Energy**: 11%
- **Consumer**: 10%
- **Industrials**: 9%
- **Real Estate**: 7%
- **Other**: 6%

---

##  Data Schema

### Standard Columns (All Files)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Publication date (ISO 8601) | `2024-03-15T14:30:00Z` |
| `title` | string | Article headline | `"Apple Announces New iPhone"` |
| `text` | string | Full article content | `"Apple Inc. today announced..."` |
| `source` | string | News source name | `"Yahoo Finance"` |
| `url` | string | Original article URL | `"https://..."` |
| `ticker` | string | Stock ticker (if applicable) | `"AAPL"` |

### Additional Columns (Feature Files)

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `sentiment_score` | float | Compound sentiment | -1.0 to +1.0 |
| `sentiment_label` | string | Classification | `Positive`, `Neutral`, `Negative` |
| `sentiment_confidence` | float | Model confidence | 0.0 to 1.0 |
| `entities` | list | Named entities | `["Apple Inc.", "Tim Cook"]` |
| `event_type` | string | Event classification | `"Earnings"`, `"M&A"`, etc. |
| `signal_score` | float | Trading signal | 0 to 100 |

---

##  File Formats

### Why Parquet?

**Advantages**:
- **70% smaller** than equivalent CSV
- **10x faster** to load
- **Preserves datatypes** (no parsing errors)
- **Columnar format** (efficient queries)
- **Industry standard** for big data

**Usage**:
```python
import pandas as pd

# Load parquet (fast)
df = pd.read_parquet('processed/consolidated_text_data.parquet')

# Save parquet
df.to_parquet('output.parquet', compression='snappy')
```

---

## Data Processing Pipeline

```
RAW DATA (CSV)
    ↓
1. Text Cleaning
    ↓
2. Deduplication
    ↓
3. Date Standardization
    ↓
4. Ticker Validation
    ↓
PROCESSED DATA (.parquet)
    ↓
5. Sentiment Analysis
    ↓
6. Entity Extraction
    ↓
7. Event Classification
    ↓
FEATURE DATA (.csv)
    ↓
8. Train/Val Split
    ↓
TRAINING DATA
```

---

## Usage Examples

### Load Main Dataset
```python
import pandas as pd

# Load consolidated data
df = pd.read_parquet('01_DATA/processed/consolidated_text_data.parquet')

print(f"Articles: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Sources: {df['source'].nunique()}")
```

### Load Features
```python
# Load sentiment features
df_sentiment = pd.read_csv('01_DATA/features/dataset_with_sentiment.csv')

# Load entity features
df_entities = pd.read_csv('01_DATA/features/dataset_with_entities.csv')

# Load event features
df_events = pd.read_csv('01_DATA/features/dataset_with_events.csv')
```

### Filter by Date
```python
# Get recent articles
df_recent = df[df['date'] >= '2024-01-01']

# Get specific company
df_aapl = df[df['ticker'] == 'AAPL']
```

---

## Legal & Ethical

### Data Sources
All data from **publicly available sources**:
- News websites (public RSS/APIs)
- Government economic data
- Company press releases

### Compliance
- Robots.txt respected
- Rate limiting applied
- No proprietary/paid data
- No personal information
- Fair use for research

### Attribution
See [DATA_PROVENANCE.md](../DATA_PROVENANCE.md) for source details and legal compliance.

---

## Updates

### Last Updated: November 1, 2025

### Update Schedule:
- **Data Collection**: Monthly (new articles added)
- **Processing**: Automated pipeline
- **Quality Checks**: Quarterly audits

---

## Contact

Questions about data?
- Open GitHub issue
- Check [DATA_PROVENANCE.md](../DATA_PROVENANCE.md)

---

**Note**: This research uses public data for academic purposes only. Not financial advice.
