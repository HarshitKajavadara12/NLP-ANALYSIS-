"""
Comprehensive Sentiment Analysis
=================================

Multi-model sentiment analysis using:
1. FinBERT (financial domain-specific)
2. VADER (lexicon-based)
3. TextBlob (pattern-based)

Ensemble approach for robust sentiment scoring.
Target: 87.3% accuracy matching institutional benchmarks
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP and Sentiment Analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm.auto import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '01_DATA'
PROCESSED_DIR = DATA_DIR / 'processed'
SENTIMENT_DIR = DATA_DIR / 'sentiment'
VIZ_DIR = DATA_DIR / 'visualizations'

# Create directories
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE SENTIMENT ANALYSIS")
print("="*80)
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print(" Step 1: Loading dataset...")

df = pd.read_csv(PROCESSED_DIR / 'financial_news_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

print(f" Loaded {len(df):,} articles")
print(f"   Companies: {df['ticker'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}\n")

# ============================================================================
# 2. SETUP FINBERT MODEL
# ============================================================================

print(" Step 2: Loading FinBERT model...")

# Check for GPU
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

print(f"   Using device: {device_name}")

# Load FinBERT
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

finbert_pipeline = pipeline(
    "sentiment-analysis",
    model=finbert_model,
    tokenizer=finbert_tokenizer,
    device=device,
    max_length=512,
    truncation=True
)

print(f" FinBERT loaded successfully\n")

# ============================================================================
# 3. FINBERT SENTIMENT ANALYSIS
# ============================================================================

print(" Step 3: Running FinBERT sentiment analysis...")
print(f"   Processing {len(df):,} articles in batches...\n")

def analyze_finbert_batch(texts, batch_size=32):
    """Analyze sentiment using FinBERT in batches"""
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT batches"):
        batch = texts[i:i+batch_size]
        
        try:
            batch_results = finbert_pipeline(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"   Error in batch {i//batch_size}: {str(e)}")
            # Add neutral results for failed batch
            results.extend([{'label': 'neutral', 'score': 0.5} for _ in batch])
    
    return results

# Run FinBERT
finbert_results = analyze_finbert_batch(df['text'].tolist())

# Extract labels and scores
df['finbert_label'] = [r['label'] for r in finbert_results]
df['finbert_score'] = [r['score'] for r in finbert_results]

# Convert to numeric sentiment (-1 to 1)
label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['finbert_sentiment'] = df['finbert_label'].map(label_map) * df['finbert_score']

print(f"\n FinBERT analysis complete")
print(f"   Positive: {(df['finbert_label'] == 'positive').sum():,} ({(df['finbert_label'] == 'positive').mean()*100:.1f}%)")
print(f"   Neutral: {(df['finbert_label'] == 'neutral').sum():,} ({(df['finbert_label'] == 'neutral').mean()*100:.1f}%)")
print(f"   Negative: {(df['finbert_label'] == 'negative').sum():,} ({(df['finbert_label'] == 'negative').mean()*100:.1f}%)")
print(f"   Average sentiment: {df['finbert_sentiment'].mean():.4f}\n")

# ============================================================================
# 4. VADER SENTIMENT ANALYSIS
# ============================================================================

print(" Step 4: Running VADER sentiment analysis...")

vader = SentimentIntensityAnalyzer()

def analyze_vader(text):
    """Analyze sentiment using VADER"""
    scores = vader.polarity_scores(str(text))
    return scores

# Run VADER
vader_results = [analyze_vader(text) for text in tqdm(df['text'], desc="VADER analysis")]

# Extract compound scores
df['vader_compound'] = [r['compound'] for r in vader_results]
df['vader_positive'] = [r['pos'] for r in vader_results]
df['vader_neutral'] = [r['neu'] for r in vader_results]
df['vader_negative'] = [r['neg'] for r in vader_results]

# Categorize
def vader_category(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_label'] = df['vader_compound'].apply(vader_category)

print(f"\n VADER analysis complete")
print(f"   Positive: {(df['vader_label'] == 'positive').sum():,} ({(df['vader_label'] == 'positive').mean()*100:.1f}%)")
print(f"   Neutral: {(df['vader_label'] == 'neutral').sum():,} ({(df['vader_label'] == 'neutral').mean()*100:.1f}%)")
print(f"   Negative: {(df['vader_label'] == 'negative').sum():,} ({(df['vader_label'] == 'negative').mean()*100:.1f}%)")
print(f"   Average compound: {df['vader_compound'].mean():.4f}\n")

# ============================================================================
# 5. TEXTBLOB SENTIMENT ANALYSIS
# ============================================================================

print(" Step 5: Running TextBlob sentiment analysis...")

def analyze_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Run TextBlob
textblob_results = [analyze_textblob(text) for text in tqdm(df['text'], desc="TextBlob analysis")]

df['textblob_polarity'] = [r[0] for r in textblob_results]
df['textblob_subjectivity'] = [r[1] for r in textblob_results]

# Categorize
def textblob_category(polarity):
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['textblob_label'] = df['textblob_polarity'].apply(textblob_category)

print(f"\n TextBlob analysis complete")
print(f"   Positive: {(df['textblob_label'] == 'positive').sum():,} ({(df['textblob_label'] == 'positive').mean()*100:.1f}%)")
print(f"   Neutral: {(df['textblob_label'] == 'neutral').sum():,} ({(df['textblob_label'] == 'neutral').mean()*100:.1f}%)")
print(f"   Negative: {(df['textblob_label'] == 'negative').sum():,} ({(df['textblob_label'] == 'negative').mean()*100:.1f}%)")
print(f"   Average polarity: {df['textblob_polarity'].mean():.4f}\n")

# ============================================================================
# 6. ENSEMBLE SENTIMENT SCORING
# ============================================================================

print(" Step 6: Creating ensemble sentiment score...")

# Weighted ensemble (FinBERT has highest weight due to domain specificity)
df['ensemble_sentiment'] = (
    df['finbert_sentiment'] * 0.5 +
    df['vader_compound'] * 0.3 +
    df['textblob_polarity'] * 0.2
)

# Ensemble category
def ensemble_category(score):
    if score > 0.15:
        return 'positive'
    elif score < -0.15:
        return 'negative'
    else:
        return 'neutral'

df['ensemble_label'] = df['ensemble_sentiment'].apply(ensemble_category)

print(f" Ensemble sentiment calculated")
print(f"   Positive: {(df['ensemble_label'] == 'positive').sum():,} ({(df['ensemble_label'] == 'positive').mean()*100:.1f}%)")
print(f"   Neutral: {(df['ensemble_label'] == 'neutral').sum():,} ({(df['ensemble_label'] == 'neutral').mean()*100:.1f}%)")
print(f"   Negative: {(df['ensemble_label'] == 'negative').sum():,} ({(df['ensemble_label'] == 'negative').mean()*100:.1f}%)")
print(f"   Average ensemble: {df['ensemble_sentiment'].mean():.4f}\n")

# ============================================================================
# 7. MODEL AGREEMENT ANALYSIS
# ============================================================================

print(" Step 7: Analyzing model agreement...")

# Check agreement between models
df['models_agree'] = (
    (df['finbert_label'] == df['vader_label']) & 
    (df['vader_label'] == df['textblob_label'])
)

agreement_rate = df['models_agree'].mean()

print(f" Model agreement analysis:")
print(f"   All 3 models agree: {df['models_agree'].sum():,} articles ({agreement_rate*100:.1f}%)")
print(f"   Disagreement: {(~df['models_agree']).sum():,} articles ({(1-agreement_rate)*100:.1f}%)\n")

# Agreement by sentiment
for sentiment in ['positive', 'neutral', 'negative']:
    subset = df[df['ensemble_label'] == sentiment]
    agree = subset['models_agree'].mean()
    print(f"   {sentiment.capitalize()} agreement: {agree*100:.1f}%")

print()

# ============================================================================
# 8. SENTIMENT BY SECTOR
# ============================================================================

print(" Step 8: Analyzing sentiment by sector...")

sector_sentiment = df.groupby('sector').agg({
    'finbert_sentiment': 'mean',
    'vader_compound': 'mean',
    'textblob_polarity': 'mean',
    'ensemble_sentiment': 'mean',
    'ticker': 'count'
}).rename(columns={'ticker': 'article_count'}).sort_values('ensemble_sentiment', ascending=False)

print(f" Sentiment by Sector (sorted by ensemble sentiment):\n")
print(sector_sentiment.round(4))
print()

# Save sector sentiment
sector_sentiment.to_csv(SENTIMENT_DIR / 'sector_sentiment.csv')
print(f" Saved: sector_sentiment.csv\n")

# ============================================================================
# 9. SENTIMENT BY COMPANY
# ============================================================================

print(" Step 9: Analyzing sentiment by company...")

company_sentiment = df.groupby(['ticker', 'company_name', 'sector']).agg({
    'finbert_sentiment': ['mean', 'std'],
    'ensemble_sentiment': ['mean', 'std'],
    'date': 'count'
}).reset_index()

company_sentiment.columns = ['ticker', 'company_name', 'sector', 
                             'finbert_mean', 'finbert_std',
                             'ensemble_mean', 'ensemble_std', 
                             'article_count']

company_sentiment = company_sentiment.sort_values('ensemble_mean', ascending=False)

print(f" Top 15 Most Positive Companies:\n")
print(company_sentiment.head(15)[['ticker', 'company_name', 'ensemble_mean', 'article_count']].to_string(index=False))

print(f"\n Top 15 Most Negative Companies:\n")
print(company_sentiment.tail(15)[['ticker', 'company_name', 'ensemble_mean', 'article_count']].to_string(index=False))
print()

# Save company sentiment
company_sentiment.to_csv(SENTIMENT_DIR / 'company_sentiment.csv', index=False)
print(f" Saved: company_sentiment.csv\n")

# ============================================================================
# 10. TEMPORAL SENTIMENT TRENDS
# ============================================================================

print(" Step 10: Analyzing temporal sentiment trends...")

# Monthly sentiment
df['year_month'] = df['date'].dt.to_period('M').astype(str)
monthly_sentiment = df.groupby('year_month').agg({
    'ensemble_sentiment': ['mean', 'std'],
    'ticker': 'count'
}).reset_index()
monthly_sentiment.columns = ['month', 'sentiment_mean', 'sentiment_std', 'article_count']

print(f" Monthly sentiment trends calculated")
print(f"   Months covered: {len(monthly_sentiment)}")
print(f"   Most positive month: {monthly_sentiment.loc[monthly_sentiment['sentiment_mean'].idxmax(), 'month']}")
print(f"   Most negative month: {monthly_sentiment.loc[monthly_sentiment['sentiment_mean'].idxmin(), 'month']}\n")

# Save temporal trends
monthly_sentiment.to_csv(SENTIMENT_DIR / 'monthly_sentiment_trends.csv', index=False)
print(f" Saved: monthly_sentiment_trends.csv\n")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

print(" Step 11: Creating visualizations...\n")

# 1. Sentiment distribution comparison
fig1 = go.Figure()

for model, col in [('FinBERT', 'finbert_sentiment'), 
                   ('VADER', 'vader_compound'),
                   ('TextBlob', 'textblob_polarity'),
                   ('Ensemble', 'ensemble_sentiment')]:
    fig1.add_trace(go.Histogram(
        x=df[col],
        name=model,
        opacity=0.7,
        nbinsx=50
    ))

fig1.update_layout(
    title='Sentiment Distribution Across Models',
    xaxis_title='Sentiment Score',
    yaxis_title='Frequency',
    barmode='overlay',
    height=500
)
fig1.write_html(VIZ_DIR / 'sentiment_distribution_comparison.html')
print(f" Saved: sentiment_distribution_comparison.html")

# 2. Sentiment by sector
fig2 = px.bar(
    sector_sentiment.reset_index(),
    x='sector',
    y='ensemble_sentiment',
    title='Average Sentiment by Sector',
    labels={'ensemble_sentiment': 'Average Ensemble Sentiment', 'sector': 'Sector'},
    color='ensemble_sentiment',
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0
)
fig2.update_layout(height=600, xaxis_tickangle=-45)
fig2.write_html(VIZ_DIR / 'sentiment_by_sector.html')
print(f" Saved: sentiment_by_sector.html")

# 3. Temporal sentiment trends
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=monthly_sentiment['month'],
    y=monthly_sentiment['sentiment_mean'],
    mode='lines+markers',
    name='Average Sentiment',
    line=dict(color='steelblue', width=2),
    error_y=dict(
        type='data',
        array=monthly_sentiment['sentiment_std'],
        visible=True
    )
))
fig3.add_hline(y=0, line_dash="dash", line_color="gray")
fig3.update_layout(
    title='Sentiment Trends Over Time (Monthly Average)',
    xaxis_title='Month',
    yaxis_title='Average Sentiment',
    height=500,
    xaxis_tickangle=-45,
    hovermode='x unified'
)
fig3.write_html(VIZ_DIR / 'sentiment_temporal_trends.html')
print(f" Saved: sentiment_temporal_trends.html")

# 4. Top 20 most positive companies
top_positive = company_sentiment.head(20)
fig4 = px.bar(
    top_positive,
    x='ensemble_mean',
    y='ticker',
    orientation='h',
    title='Top 20 Most Positive Companies',
    labels={'ensemble_mean': 'Average Sentiment', 'ticker': 'Ticker'},
    color='ensemble_mean',
    color_continuous_scale='Greens',
    hover_data=['company_name', 'article_count']
)
fig4.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
fig4.write_html(VIZ_DIR / 'top_positive_companies.html')
print(f" Saved: top_positive_companies.html")

# 5. Model correlation
model_corr = df[['finbert_sentiment', 'vader_compound', 'textblob_polarity', 'ensemble_sentiment']].corr()
fig5 = px.imshow(
    model_corr,
    text_auto='.3f',
    title='Sentiment Model Correlation Matrix',
    color_continuous_scale='RdBu_r',
    aspect='auto',
    labels=dict(x="Model", y="Model", color="Correlation")
)
fig5.update_layout(height=500)
fig5.write_html(VIZ_DIR / 'sentiment_model_correlation.html')
print(f" Saved: sentiment_model_correlation.html\n")

# ============================================================================
# 12. SAVE ENHANCED DATASET
# ============================================================================

print(" Step 12: Saving enhanced dataset with sentiment scores...")

# Select columns to save
sentiment_columns = [
    'ticker', 'company_name', 'sector', 'industry', 'date',
    'title', 'text', 'source',
    'finbert_label', 'finbert_score', 'finbert_sentiment',
    'vader_compound', 'vader_positive', 'vader_neutral', 'vader_negative', 'vader_label',
    'textblob_polarity', 'textblob_subjectivity', 'textblob_label',
    'ensemble_sentiment', 'ensemble_label',
    'models_agree',
    'word_count', 'char_count', 'sentence_count',
    'year', 'month', 'day', 'hour', 'is_weekend', 'is_market_hours'
]

df_sentiment = df[sentiment_columns].copy()

# Save
output_file = PROCESSED_DIR / 'dataset_with_sentiment.csv'
df_sentiment.to_csv(output_file, index=False)

print(f" Enhanced dataset saved to {output_file}")
print(f"   Rows: {len(df_sentiment):,}")
print(f"   Columns: {len(df_sentiment.columns)}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

# ============================================================================
# 13. GENERATE SUMMARY REPORT
# ============================================================================

print(" Step 13: Generating sentiment analysis summary...\n")

summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'total_articles': int(len(df)),
        'companies': int(df['ticker'].nunique()),
        'sectors': int(df['sector'].nunique())
    },
    'finbert': {
        'positive_pct': float((df['finbert_label'] == 'positive').mean() * 100),
        'neutral_pct': float((df['finbert_label'] == 'neutral').mean() * 100),
        'negative_pct': float((df['finbert_label'] == 'negative').mean() * 100),
        'avg_sentiment': float(df['finbert_sentiment'].mean()),
        'std_sentiment': float(df['finbert_sentiment'].std())
    },
    'vader': {
        'positive_pct': float((df['vader_label'] == 'positive').mean() * 100),
        'neutral_pct': float((df['vader_label'] == 'neutral').mean() * 100),
        'negative_pct': float((df['vader_label'] == 'negative').mean() * 100),
        'avg_compound': float(df['vader_compound'].mean()),
        'std_compound': float(df['vader_compound'].std())
    },
    'textblob': {
        'positive_pct': float((df['textblob_label'] == 'positive').mean() * 100),
        'neutral_pct': float((df['textblob_label'] == 'neutral').mean() * 100),
        'negative_pct': float((df['textblob_label'] == 'negative').mean() * 100),
        'avg_polarity': float(df['textblob_polarity'].mean()),
        'std_polarity': float(df['textblob_polarity'].std())
    },
    'ensemble': {
        'positive_pct': float((df['ensemble_label'] == 'positive').mean() * 100),
        'neutral_pct': float((df['ensemble_label'] == 'neutral').mean() * 100),
        'negative_pct': float((df['ensemble_label'] == 'negative').mean() * 100),
        'avg_sentiment': float(df['ensemble_sentiment'].mean()),
        'std_sentiment': float(df['ensemble_sentiment'].std())
    },
    'model_agreement': {
        'agreement_rate': float(agreement_rate * 100),
        'disagreement_rate': float((1 - agreement_rate) * 100)
    },
    'sector_sentiment': {
        'most_positive': sector_sentiment.index[0],
        'most_negative': sector_sentiment.index[-1],
        'by_sector': sector_sentiment['ensemble_sentiment'].to_dict()
    },
    'company_sentiment': {
        'most_positive': {
            'ticker': company_sentiment.iloc[0]['ticker'],
            'company_name': company_sentiment.iloc[0]['company_name'],
            'sentiment': float(company_sentiment.iloc[0]['ensemble_mean'])
        },
        'most_negative': {
            'ticker': company_sentiment.iloc[-1]['ticker'],
            'company_name': company_sentiment.iloc[-1]['company_name'],
            'sentiment': float(company_sentiment.iloc[-1]['ensemble_mean'])
        }
    }
}

# Save summary
summary_file = SENTIMENT_DIR / 'sentiment_analysis_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f" Saved: sentiment_analysis_summary.json")

# ============================================================================
# 14. FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("SENTIMENT ANALYSIS COMPLETE")
print("="*80)

print(f"\n Analysis Summary:")
print(f"   Articles analyzed: {summary['dataset']['total_articles']:,}")
print(f"   Models used: FinBERT, VADER, TextBlob, Ensemble")
print(f"   Model agreement rate: {summary['model_agreement']['agreement_rate']:.1f}%")

print(f"\n Ensemble Sentiment Distribution:")
print(f"   Positive: {summary['ensemble']['positive_pct']:.1f}%")
print(f"   Neutral: {summary['ensemble']['neutral_pct']:.1f}%")
print(f"   Negative: {summary['ensemble']['negative_pct']:.1f}%")
print(f"   Average: {summary['ensemble']['avg_sentiment']:.4f}")

print(f"\n Sector Insights:")
print(f"   Most positive: {summary['sector_sentiment']['most_positive']}")
print(f"   Most negative: {summary['sector_sentiment']['most_negative']}")

print(f"\n Company Insights:")
print(f"   Most positive: {summary['company_sentiment']['most_positive']['ticker']} - {summary['company_sentiment']['most_positive']['company_name']}")
print(f"                  Sentiment: {summary['company_sentiment']['most_positive']['sentiment']:.4f}")
print(f"   Most negative: {summary['company_sentiment']['most_negative']['ticker']} - {summary['company_sentiment']['most_negative']['company_name']}")
print(f"                  Sentiment: {summary['company_sentiment']['most_negative']['sentiment']:.4f}")

print(f"\n Output Files:")
print(f"   1. {output_file}")
print(f"   2. {summary_file}")
print(f"   3. {SENTIMENT_DIR / 'sector_sentiment.csv'}")
print(f"   4. {SENTIMENT_DIR / 'company_sentiment.csv'}")
print(f"   5. {SENTIMENT_DIR / 'monthly_sentiment_trends.csv'}")
print(f"   6. {len(list(VIZ_DIR.glob('sentiment_*.html')))} visualizations")

print(f"\n Ready for Step 4: Named Entity Recognition")
print("="*80)
