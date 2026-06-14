"""
Advanced Feature Engineering - Comprehensive
============================================

Creates 47+ predictive features across 8 categories:
1. Sentiment features (9)
2. Entity features (5)
3. Event features (6)
4. Topic features (4)
5. Temporal features (8)
6. News velocity (6)
7. Text complexity (5)
8. Composite features (4)

Target: High-quality feature matrix for ML modeling
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from tqdm.auto import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('seaborn-v0_8-darkgrid')

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '01_DATA'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features'
VIZ_DIR = DATA_DIR / 'visualizations'

# Create directories
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ADVANCED FEATURE ENGINEERING - COMPREHENSIVE")
print("="*80)
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print(" Step 1: Loading dataset with topics...")

df = pd.read_csv(PROCESSED_DIR / 'dataset_with_topics.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort by ticker and date
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

print(f" Loaded {len(df):,} articles")
print(f"   Companies: {df['ticker'].nunique()}")
print(f"   Features (before engineering): {df.shape[1]}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}\n")

# ============================================================================
# 2. SENTIMENT FEATURES
# ============================================================================

print(" Step 2: Engineering sentiment features...")

# Group by ticker for time-series features
df['sentiment_momentum'] = df.groupby('ticker')['finbert_sentiment'].diff()
df['sentiment_ma_3d'] = df.groupby('ticker')['finbert_sentiment'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
df['sentiment_ma_7d'] = df.groupby('ticker')['finbert_sentiment'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
df['sentiment_ma_14d'] = df.groupby('ticker')['finbert_sentiment'].rolling(window=14, min_periods=1).mean().reset_index(0, drop=True)
df['sentiment_volatility'] = df.groupby('ticker')['finbert_sentiment'].rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
df['sentiment_range_7d'] = df.groupby('ticker')['finbert_sentiment'].rolling(window=7, min_periods=1).apply(lambda x: x.max() - x.min()).reset_index(0, drop=True)
df['sentiment_acceleration'] = df.groupby('ticker')['sentiment_momentum'].diff()
df['sentiment_extremity'] = df['finbert_sentiment'].abs()
df['sentiment_consensus'] = (df['finbert_sentiment'] * df['ensemble_sentiment']).abs()

sentiment_features = [
    'sentiment_momentum', 'sentiment_ma_3d', 'sentiment_ma_7d', 'sentiment_ma_14d',
    'sentiment_volatility', 'sentiment_range_7d', 'sentiment_acceleration',
    'sentiment_extremity', 'sentiment_consensus'
]

print(f" Created {len(sentiment_features)} sentiment features")
print(f"   {', '.join(sentiment_features[:5])}, ...\n")

# ============================================================================
# 3. ENTITY FEATURES
# ============================================================================

print(" Step 3: Engineering entity features...")

# Entity density and diversity
df['entity_density'] = df['num_entities'] / df['word_count']
df['entity_diversity'] = (df['num_orgs'] + df['num_persons'] + df['num_locations']) / (df['num_entities'] + 1)
df['org_prominence'] = df['num_orgs'] / (df['num_entities'] + 1)
df['entity_freq_7d'] = df.groupby('ticker')['num_entities'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
df['entity_momentum'] = df.groupby('ticker')['num_entities'].diff()

entity_features = [
    'entity_density', 'entity_diversity', 'org_prominence',
    'entity_freq_7d', 'entity_momentum'
]

print(f" Created {len(entity_features)} entity features")
print(f"   {', '.join(entity_features)}\n")

# ============================================================================
# 4. EVENT FEATURES
# ============================================================================

print(" Step 4: Engineering event features...")

df['event_density'] = df['num_events'] / df['word_count']
df['event_freq_7d'] = df.groupby('ticker')['num_events'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
df['event_momentum'] = df.groupby('ticker')['num_events'].diff()
df['event_impact_avg_7d'] = df.groupby('ticker')['event_impact_score'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
df['sentiment_event_alignment'] = df['finbert_sentiment'] * df['event_impact_score']

event_features = [
    'event_density', 'has_high_impact_event', 'event_freq_7d',
    'event_momentum', 'event_impact_avg_7d', 'sentiment_event_alignment'
]

print(f" Created {len(event_features)} event features")
print(f"   {', '.join(event_features)}\n")

# ============================================================================
# 5. TOPIC FEATURES
# ============================================================================

print(" Step 5: Engineering topic features...")

# Topic entropy (distribution across topics)
topic_prob_cols = [col for col in df.columns if col.startswith('topic_') and col.endswith('_prob')]
if topic_prob_cols:
    topic_probs = df[topic_prob_cols].values
    df['topic_entropy'] = -np.sum(topic_probs * np.log(topic_probs + 1e-10), axis=1)
else:
    df['topic_entropy'] = 0

df['topic_dominance'] = df['topic_probability']
df['topic_shift'] = df.groupby('ticker')['dominant_topic'].apply(lambda x: (x != x.shift()).astype(int)).reset_index(0, drop=True)
df['topic_stability'] = 1 - df.groupby('ticker')['topic_shift'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

topic_features = [
    'topic_entropy', 'topic_dominance', 'topic_shift', 'topic_stability'
]

print(f" Created {len(topic_features)} topic features")
print(f"   {', '.join(topic_features)}\n")

# ============================================================================
# 6. TEMPORAL FEATURES
# ============================================================================

print(" Step 6: Engineering temporal features...")

df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16) & (df['is_weekend'] == 0)).astype(int)

# Days since last article
df['days_since_last'] = df.groupby('ticker')['date'].diff().dt.days

# Recency score (exponential decay)
max_days = df['days_since_last'].max()
df['recency_score'] = np.exp(-df['days_since_last'] / 7)

# Lagged sentiment
df['sentiment_lag_1'] = df.groupby('ticker')['finbert_sentiment'].shift(1)
df['sentiment_lag_7'] = df.groupby('ticker')['finbert_sentiment'].shift(7)

temporal_features = [
    'hour', 'day_of_week', 'is_weekend', 'is_market_hours',
    'days_since_last', 'recency_score', 'sentiment_lag_1', 'sentiment_lag_7'
]

print(f" Created {len(temporal_features)} temporal features")
print(f"   {', '.join(temporal_features[:5])}, ...\n")

# ============================================================================
# 7. NEWS VELOCITY FEATURES
# ============================================================================

print(" Step 7: Engineering news velocity features...")

# Count articles in time windows
df['article_count_1d'] = df.groupby('ticker')['date'].transform(
    lambda x: x.rolling('1D', on=x).count()
)
df['article_count_7d'] = df.groupby('ticker')['date'].transform(
    lambda x: x.rolling('7D', on=x).count()
)
df['article_count_30d'] = df.groupby('ticker')['date'].transform(
    lambda x: x.rolling('30D', on=x).count()
)

# News acceleration
df['news_acceleration'] = df.groupby('ticker')['article_count_7d'].diff()

# News burst detection
df['news_burst'] = (df['article_count_1d'] > df['article_count_7d'].quantile(0.95)).astype(int)

# Coverage consistency
df['coverage_consistency'] = df['article_count_7d'] / (df['article_count_30d'] + 1)

velocity_features = [
    'article_count_1d', 'article_count_7d', 'article_count_30d',
    'news_acceleration', 'news_burst', 'coverage_consistency'
]

print(f" Created {len(velocity_features)} news velocity features")
print(f"   {', '.join(velocity_features)}\n")

# ============================================================================
# 8. TEXT COMPLEXITY FEATURES
# ============================================================================

print(" Step 8: Engineering text complexity features...")

# Basic readability metrics
df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
df['sentence_count'] = df['text'].str.count(r'[.!?]') + 1
df['avg_sentence_length'] = df['word_count'] / df['sentence_count']

# Flesch Reading Ease approximation
df['readability_score'] = 206.835 - 1.015 * df['avg_sentence_length'] - 84.6 * (df['avg_word_length'] / df['word_count'])

complexity_features = [
    'readability_score', 'word_count', 'avg_word_length',
    'sentence_count', 'avg_sentence_length'
]

print(f" Created {len(complexity_features)} text complexity features")
print(f"   {', '.join(complexity_features)}\n")

# ============================================================================
# 9. COMPOSITE FEATURES
# ============================================================================

print(" Step 9: Engineering composite features...")

# News importance score
df['news_importance'] = (
    df['event_impact_score'] * 0.4 +
    df['sentiment_extremity'] * 0.3 +
    df['entity_density'] * 100 * 0.3
)

# Attention score
df['attention_score'] = (
    df['article_count_7d'] * 0.5 +
    df['news_burst'] * 10 +
    df['has_high_impact_event'] * 5
)

# Signal strength
df['signal_strength'] = (
    df['sentiment_extremity'] * df['sentiment_consensus'] * 
    (1 + df['event_impact_score']) * (1 + df['topic_dominance'])
)

# Market relevance
df['market_relevance'] = (
    df['is_market_hours'] * 0.3 +
    (1 - df['is_weekend']) * 0.2 +
    df['entity_density'] * 50 * 0.3 +
    df['news_importance'] / 10 * 0.2
)

composite_features = [
    'news_importance', 'attention_score', 'signal_strength', 'market_relevance'
]

print(f" Created {len(composite_features)} composite features")
print(f"   {', '.join(composite_features)}\n")

# ============================================================================
# 10. HANDLE MISSING VALUES
# ============================================================================

print(" Step 10: Handling missing values...")

# Count missing values before
missing_before = df.isnull().sum().sum()

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df[col].isnull().any():
        # Fill with median by ticker
        df[col] = df.groupby('ticker')[col].transform(lambda x: x.fillna(x.median()))
        
        # Fill remaining with global median
        df[col] = df[col].fillna(df[col].median())
        
        # Fill any remaining with 0
        df[col] = df[col].fillna(0)

# Replace infinity with NaN then 0
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

missing_after = df.isnull().sum().sum()

print(f" Missing values handled")
print(f"   Before: {missing_before:,}")
print(f"   After: {missing_after:,}\n")

# ============================================================================
# 11. FEATURE SUMMARY
# ============================================================================

print(" Step 11: Generating feature summary...\n")

all_features = (
    sentiment_features + entity_features + event_features +
    topic_features + temporal_features + velocity_features +
    complexity_features + composite_features
)

print("="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)
print(f"\nTotal features created: {len(all_features)}")
print(f"\nFeature Categories:")
print(f"   1. Sentiment features: {len(sentiment_features)}")
print(f"   2. Entity features: {len(entity_features)}")
print(f"   3. Event features: {len(event_features)}")
print(f"   4. Topic features: {len(topic_features)}")
print(f"   5. Temporal features: {len(temporal_features)}")
print(f"   6. News velocity features: {len(velocity_features)}")
print(f"   7. Text complexity features: {len(complexity_features)}")
print(f"   8. Composite features: {len(composite_features)}")
print("="*80)
print()

# Save feature metadata
feature_metadata = {
    'total_features': len(all_features),
    'feature_categories': {
        'sentiment': sentiment_features,
        'entity': entity_features,
        'event': event_features,
        'topic': topic_features,
        'temporal': temporal_features,
        'velocity': velocity_features,
        'complexity': complexity_features,
        'composite': composite_features
    },
    'feature_list': all_features
}

with open(FEATURES_DIR / 'feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print(f" Saved: feature_metadata.json\n")

# ============================================================================
# 12. FEATURE STATISTICS
# ============================================================================

print(" Step 12: Calculating feature statistics...")

feature_stats = df[all_features].describe().T
feature_stats['skewness'] = df[all_features].skew()
feature_stats['kurtosis'] = df[all_features].kurtosis()

print(f" Feature statistics calculated")
print(f"\n   Sample statistics (first 10 features):")
print(feature_stats.head(10)[['mean', 'std', 'min', 'max']].round(4))
print()

# Save
feature_stats.to_csv(FEATURES_DIR / 'feature_statistics.csv')
print(f" Saved: feature_statistics.csv\n")

# ============================================================================
# 13. FEATURE CORRELATION ANALYSIS
# ============================================================================

print(" Step 13: Analyzing feature correlations...")

# Select subset of key features for correlation
key_features = [
    'finbert_sentiment', 'sentiment_momentum', 'sentiment_extremity',
    'num_entities', 'entity_density', 'num_events', 'event_impact_score',
    'topic_dominance', 'article_count_7d', 'news_importance'
]

corr_matrix = df[key_features].corr()

print(f" Correlation matrix calculated")
print(f"   Features analyzed: {len(key_features)}")
print(f"   Highest correlation pairs:")

# Find top correlations
corr_pairs = []
for i in range(len(key_features)):
    for j in range(i+1, len(key_features)):
        corr_pairs.append({
            'feature1': key_features[i],
            'feature2': key_features[j],
            'correlation': corr_matrix.iloc[i, j]
        })

corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('correlation', key=abs, ascending=False)
print(corr_pairs_df.head(5).to_string(index=False))
print()

# Visualization
fig = px.imshow(
    corr_matrix,
    text_auto='.2f',
    title='Feature Correlation Matrix (Key Features)',
    color_continuous_scale='RdBu_r',
    aspect='auto',
    color_continuous_midpoint=0
)
fig.update_layout(height=700)
fig.write_html(VIZ_DIR / 'feature_correlation_matrix.html')
print(f" Saved: feature_correlation_matrix.html\n")

# ============================================================================
# 14. SAVE FINAL FEATURE MATRIX
# ============================================================================

print(" Step 14: Saving final feature matrix...")

# Select columns for final dataset
final_columns = [
    # Identifiers
    'ticker', 'company_name', 'sector', 'industry', 'date',
    'title', 'text', 'source',
    
    # Base features
    'finbert_sentiment', 'ensemble_sentiment',
    'num_entities', 'num_orgs', 'num_persons', 'num_locations',
    'num_events', 'primary_event', 'event_impact_score',
    'dominant_topic', 'topic_probability',
    'word_count', 'year', 'month'
]

# Add all engineered features
final_columns.extend(all_features)

# Remove duplicates
final_columns = list(dict.fromkeys(final_columns))

df_final = df[final_columns].copy()

# Save
output_file = FEATURES_DIR / 'final_feature_matrix.csv'
df_final.to_csv(output_file, index=False)

print(f" Final feature matrix saved to {output_file}")
print(f"   Rows: {len(df_final):,}")
print(f"   Columns: {len(df_final.columns)}")
print(f"   Features: {len(all_features)}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

# ============================================================================
# 15. FINAL REPORT
# ============================================================================

print("="*80)
print("ADVANCED FEATURE ENGINEERING COMPLETE")
print("="*80)

print(f"\n Feature Engineering Summary:")
print(f"   Articles processed: {len(df_final):,}")
print(f"   Total features created: {len(all_features)}")
print(f"   Missing values: {missing_after}")

print(f"\n Feature Categories Breakdown:")
for category, features in feature_metadata['feature_categories'].items():
    print(f"   {category.capitalize()}: {len(features)} features")

print(f"\n Key Feature Correlations:")
for idx, row in corr_pairs_df.head(3).iterrows():
    print(f"   {row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")

print(f"\n Output Files:")
print(f"   1. {output_file}")
print(f"   2. {FEATURES_DIR / 'feature_metadata.json'}")
print(f"   3. {FEATURES_DIR / 'feature_statistics.csv'}")
print(f"   4. {VIZ_DIR / 'feature_correlation_matrix.html'}")

print(f"\n Ready for Step 8: Ensemble Modeling Strategies")
print("="*80)
