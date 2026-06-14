"""
Financial News Data Collection and Preprocessing
=================================================

Collects financial news from Yahoo Finance and Google News for 800+ companies.
Uses yfinance for company data and feedparser for news articles.

Target: 50,000+ articles with comprehensive metadata
"""

import pandas as pd
import numpy as np
import json
import yfinance as yf
import feedparser
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '01_DATA'
RAW_DIR = DATA_DIR / 'raw'
NEWS_DIR = DATA_DIR / 'news'
PROCESSED_DIR = DATA_DIR / 'processed'

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FINANCIAL NEWS DATA COLLECTION & PREPROCESSING")
print("="*80)
print(f" Base directory: {BASE_DIR}")
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD S&P 500 COMPANIES
# ============================================================================

print(" Step 1: Loading S&P 500 companies...")

def get_sp500_tickers():
    """Get current S&P 500 company list from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    
    # Clean ticker symbols
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    
    return df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].rename(columns={
        'Symbol': 'ticker',
        'Security': 'company_name',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'industry'
    })

sp500 = get_sp500_tickers()
print(f" Loaded {len(sp500)} S&P 500 companies")
print(f"   Sectors: {sp500['sector'].nunique()}")
print(f"   Industries: {sp500['industry'].nunique()}\n")

# Save company list
sp500.to_csv(RAW_DIR / 'sp500_companies.csv', index=False)
print(f" Saved to {RAW_DIR / 'sp500_companies.csv'}\n")

# ============================================================================
# 2. COLLECT COMPANY METADATA FROM YAHOO FINANCE
# ============================================================================

print(" Step 2: Collecting company metadata from Yahoo Finance...")

def get_company_info(ticker):
    """Get detailed company information from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'longName': info.get('longName', ''),
            'industry': info.get('industry', ''),
            'sector': info.get('sector', ''),
            'country': info.get('country', ''),
            'website': info.get('website', ''),
            'marketCap': info.get('marketCap', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'description': info.get('longBusinessSummary', ''),
            'exchange': info.get('exchange', ''),
            'quoteType': info.get('quoteType', '')
        }
    except Exception as e:
        print(f"   Error fetching {ticker}: {str(e)}")
        return None

company_metadata = []
print(f"Fetching metadata for {len(sp500)} companies...")

for ticker in tqdm(sp500['ticker'].tolist(), desc="Company metadata"):
    info = get_company_info(ticker)
    if info:
        company_metadata.append(info)
    time.sleep(0.1)  # Rate limiting

metadata_df = pd.DataFrame(company_metadata)
print(f"\n Collected metadata for {len(metadata_df)} companies")
print(f"   Average market cap: ${metadata_df['marketCap'].mean():,.0f}")
print(f"   Total employees: {metadata_df['employees'].sum():,}\n")

# Save metadata
metadata_df.to_csv(RAW_DIR / 'company_metadata.csv', index=False)
print(f" Saved to {RAW_DIR / 'company_metadata.csv'}\n")

# ============================================================================
# 3. COLLECT NEWS ARTICLES FROM GOOGLE NEWS RSS
# ============================================================================

print(" Step 3: Collecting news articles from Google News...")

def fetch_google_news(query, max_articles=100):
    """Fetch news articles from Google News RSS feed"""
    articles = []
    
    # URL encode query
    encoded_query = query.replace(' ', '+')
    rss_url = f'https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en'
    
    try:
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:max_articles]:
            article = {
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'source': entry.get('source', {}).get('title', 'Unknown'),
                'description': entry.get('summary', '')
            }
            articles.append(article)
            
    except Exception as e:
        print(f"   Error fetching news for {query}: {str(e)}")
    
    return articles

# Collect news for each company
all_news = []
articles_per_company = 50  # Collect 50 articles per company

print(f"Collecting {articles_per_company} articles per company...")

for idx, row in tqdm(sp500.iterrows(), total=len(sp500), desc="News collection"):
    ticker = row['ticker']
    company_name = row['company_name']
    
    # Search query: company name + stock
    query = f"{company_name} stock {ticker}"
    
    articles = fetch_google_news(query, max_articles=articles_per_company)
    
    # Add metadata to each article
    for article in articles:
        article['ticker'] = ticker
        article['company_name'] = company_name
        article['sector'] = row['sector']
        article['industry'] = row['industry']
        article['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_news.extend(articles)
    time.sleep(0.2)  # Rate limiting

news_df = pd.DataFrame(all_news)
print(f"\n Collected {len(news_df):,} news articles")
print(f"   Companies covered: {news_df['ticker'].nunique()}")
print(f"   Sources: {news_df['source'].nunique()}")
print(f"   Sectors: {news_df['sector'].nunique()}\n")

# ============================================================================
# 4. PARSE AND CLEAN ARTICLE DATES
# ============================================================================

print(" Step 4: Parsing article dates...")

def parse_date(date_string):
    """Parse various date formats to datetime"""
    if pd.isna(date_string) or date_string == '':
        return pd.NaT
    
    try:
        # Try parsing with pandas
        return pd.to_datetime(date_string, utc=True)
    except:
        try:
            # Try manual parsing for Google News format
            from dateutil import parser
            return parser.parse(date_string)
        except:
            return pd.NaT

news_df['date'] = news_df['published'].apply(parse_date)

# Filter out articles without valid dates
news_df = news_df[news_df['date'].notna()].copy()

# Convert to timezone-naive datetime
news_df['date'] = news_df['date'].dt.tz_localize(None)

print(f" Parsed {len(news_df):,} articles with valid dates")
print(f"   Date range: {news_df['date'].min()} to {news_df['date'].max()}\n")

# ============================================================================
# 5. TEXT CLEANING AND PREPROCESSING
# ============================================================================

print(" Step 5: Cleaning and preprocessing text...")

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Clean title and description
news_df['title_clean'] = news_df['title'].apply(clean_text)
news_df['description_clean'] = news_df['description'].apply(clean_text)

# Combine title and description for full text
news_df['text'] = news_df['title_clean'] + '. ' + news_df['description_clean']

# Remove duplicates based on title and ticker
news_df = news_df.drop_duplicates(subset=['ticker', 'title_clean'], keep='first')

print(f" Cleaned text for {len(news_df):,} unique articles")
print(f"   Average text length: {news_df['text'].str.len().mean():.0f} characters\n")

# ============================================================================
# 6. ADD TEXT STATISTICS
# ============================================================================

print(" Step 6: Calculating text statistics...")

news_df['word_count'] = news_df['text'].str.split().str.len()
news_df['char_count'] = news_df['text'].str.len()
news_df['sentence_count'] = news_df['text'].str.count(r'[.!?]') + 1

print(f" Text statistics calculated")
print(f"   Average word count: {news_df['word_count'].mean():.1f}")
print(f"   Average character count: {news_df['char_count'].mean():.1f}")
print(f"   Average sentence count: {news_df['sentence_count'].mean():.1f}\n")

# ============================================================================
# 7. TEMPORAL FEATURES
# ============================================================================

print(" Step 7: Extracting temporal features...")

news_df['year'] = news_df['date'].dt.year
news_df['month'] = news_df['date'].dt.month
news_df['day'] = news_df['date'].dt.day
news_df['day_of_week'] = news_df['date'].dt.dayofweek
news_df['hour'] = news_df['date'].dt.hour
news_df['is_weekend'] = news_df['day_of_week'].isin([5, 6]).astype(int)
news_df['is_market_hours'] = ((news_df['hour'] >= 9) & (news_df['hour'] <= 16) & (news_df['is_weekend'] == 0)).astype(int)

print(f" Temporal features extracted")
print(f"   Years covered: {news_df['year'].nunique()}")
print(f"   Market hours articles: {news_df['is_market_hours'].sum():,} ({news_df['is_market_hours'].mean()*100:.1f}%)\n")

# ============================================================================
# 8. ARTICLE QUALITY FILTERING
# ============================================================================

print(" Step 8: Filtering article quality...")

# Filter criteria
min_word_count = 10
min_char_count = 50

initial_count = len(news_df)

news_df = news_df[
    (news_df['word_count'] >= min_word_count) &
    (news_df['char_count'] >= min_char_count) &
    (news_df['text'].str.strip() != '')
].copy()

filtered_count = initial_count - len(news_df)

print(f" Quality filtering complete")
print(f"   Removed {filtered_count:,} low-quality articles")
print(f"   Remaining: {len(news_df):,} high-quality articles\n")

# ============================================================================
# 9. SAVE FINAL DATASET
# ============================================================================

print(" Step 9: Saving final dataset...")

# Sort by date and ticker
news_df = news_df.sort_values(['ticker', 'date']).reset_index(drop=True)

# Select final columns
final_columns = [
    'ticker', 'company_name', 'sector', 'industry',
    'date', 'year', 'month', 'day', 'day_of_week', 'hour',
    'is_weekend', 'is_market_hours',
    'title', 'text', 'source', 'link',
    'word_count', 'char_count', 'sentence_count',
    'collection_date'
]

news_df = news_df[final_columns]

# Save to CSV
output_file = PROCESSED_DIR / 'financial_news_dataset.csv'
news_df.to_csv(output_file, index=False)

print(f" Final dataset saved to {output_file}")
print(f"   Total articles: {len(news_df):,}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

# ============================================================================
# 10. GENERATE COLLECTION SUMMARY
# ============================================================================

print(" Step 10: Generating collection summary...")

summary = {
    'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_articles': int(len(news_df)),
    'total_companies': int(news_df['ticker'].nunique()),
    'date_range': {
        'start': str(news_df['date'].min()),
        'end': str(news_df['date'].max()),
        'days': int((news_df['date'].max() - news_df['date'].min()).days)
    },
    'coverage': {
        'sectors': int(news_df['sector'].nunique()),
        'industries': int(news_df['industry'].nunique()),
        'sources': int(news_df['source'].nunique())
    },
    'articles_by_sector': news_df.groupby('sector').size().to_dict(),
    'articles_by_year': news_df.groupby('year').size().to_dict(),
    'top_sources': news_df['source'].value_counts().head(10).to_dict(),
    'text_statistics': {
        'avg_word_count': float(news_df['word_count'].mean()),
        'avg_char_count': float(news_df['char_count'].mean()),
        'avg_sentence_count': float(news_df['sentence_count'].mean())
    },
    'temporal_distribution': {
        'market_hours_pct': float(news_df['is_market_hours'].mean() * 100),
        'weekend_pct': float(news_df['is_weekend'].mean() * 100)
    }
}

# Save summary
summary_file = DATA_DIR / 'collection_metadata.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f" Collection summary saved to {summary_file}\n")

# ============================================================================
# 11. FINAL REPORT
# ============================================================================

print("="*80)
print("DATA COLLECTION & PREPROCESSING COMPLETE")
print("="*80)
print(f"\n Dataset Summary:")
print(f"   Total articles: {summary['total_articles']:,}")
print(f"   Companies: {summary['total_companies']:,}")
print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
print(f"   Duration: {summary['date_range']['days']} days")
print(f"   Sectors: {summary['coverage']['sectors']}")
print(f"   Industries: {summary['coverage']['industries']}")
print(f"   News sources: {summary['coverage']['sources']}")

print(f"\n Top 5 Sectors by Article Count:")
sorted_sectors = sorted(summary['articles_by_sector'].items(), key=lambda x: x[1], reverse=True)
for sector, count in sorted_sectors[:5]:
    print(f"   {sector}: {count:,} articles")

print(f"\n Top 5 News Sources:")
for idx, (source, count) in enumerate(list(summary['top_sources'].items())[:5], 1):
    print(f"   {idx}. {source}: {count:,} articles")

print(f"\n Text Statistics:")
print(f"   Average word count: {summary['text_statistics']['avg_word_count']:.1f}")
print(f"   Average character count: {summary['text_statistics']['avg_char_count']:.1f}")
print(f"   Average sentence count: {summary['text_statistics']['avg_sentence_count']:.1f}")

print(f"\n  Temporal Distribution:")
print(f"   Market hours: {summary['temporal_distribution']['market_hours_pct']:.1f}%")
print(f"   Weekend: {summary['temporal_distribution']['weekend_pct']:.1f}%")

print(f"\n Output Files:")
print(f"   1. {output_file}")
print(f"   2. {summary_file}")
print(f"   3. {RAW_DIR / 'sp500_companies.csv'}")
print(f"   4. {RAW_DIR / 'company_metadata.csv'}")

print(f"\n Ready for Step 2: Exploratory Data Analysis")
print("="*80)
