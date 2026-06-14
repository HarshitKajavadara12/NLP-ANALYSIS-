"""
Exploratory Data Analysis for Financial News
=============================================

Comprehensive EDA analyzing temporal patterns, coverage distribution,
text characteristics, and sector-specific insights.

Generates 15+ visualizations and statistical reports.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '01_DATA'
PROCESSED_DIR = DATA_DIR / 'processed'
ANALYSIS_DIR = DATA_DIR / 'analysis'
VIZ_DIR = DATA_DIR / 'visualizations'

# Create directories
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - FINANCIAL NEWS")
print("="*80)
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print(" Step 1: Loading dataset...")

df = pd.read_csv(PROCESSED_DIR / 'financial_news_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

print(f" Loaded {len(df):,} articles")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Companies: {df['ticker'].nunique()}")
print(f"   Columns: {df.shape[1]}\n")

# ============================================================================
# 2. OVERVIEW STATISTICS
# ============================================================================

print(" Step 2: Calculating overview statistics...\n")

overview_stats = {
    'dataset_size': {
        'total_articles': int(len(df)),
        'unique_companies': int(df['ticker'].nunique()),
        'unique_sectors': int(df['sector'].nunique()),
        'unique_industries': int(df['industry'].nunique()),
        'unique_sources': int(df['source'].nunique())
    },
    'temporal_coverage': {
        'start_date': str(df['date'].min()),
        'end_date': str(df['date'].max()),
        'days_covered': int((df['date'].max() - df['date'].min()).days),
        'years_covered': int(df['year'].nunique()),
        'months_covered': int(df.groupby(['year', 'month']).ngroups)
    },
    'text_statistics': {
        'avg_word_count': float(df['word_count'].mean()),
        'median_word_count': float(df['word_count'].median()),
        'std_word_count': float(df['word_count'].std()),
        'min_word_count': int(df['word_count'].min()),
        'max_word_count': int(df['word_count'].max()),
        'avg_char_count': float(df['char_count'].mean()),
        'avg_sentence_count': float(df['sentence_count'].mean())
    },
    'distribution': {
        'market_hours_pct': float(df['is_market_hours'].mean() * 100),
        'weekend_pct': float(df['is_weekend'].mean() * 100),
        'weekday_pct': float((1 - df['is_weekend'].mean()) * 100)
    }
}

print(" OVERVIEW STATISTICS:")
print("="*80)
print(f"\nDataset Size:")
print(f"   Total articles: {overview_stats['dataset_size']['total_articles']:,}")
print(f"   Unique companies: {overview_stats['dataset_size']['unique_companies']:,}")
print(f"   Sectors: {overview_stats['dataset_size']['unique_sectors']}")
print(f"   Industries: {overview_stats['dataset_size']['unique_industries']}")
print(f"   News sources: {overview_stats['dataset_size']['unique_sources']}")

print(f"\nTemporal Coverage:")
print(f"   Date range: {overview_stats['temporal_coverage']['start_date']} to {overview_stats['temporal_coverage']['end_date']}")
print(f"   Days: {overview_stats['temporal_coverage']['days_covered']:,}")
print(f"   Years: {overview_stats['temporal_coverage']['years_covered']}")
print(f"   Months: {overview_stats['temporal_coverage']['months_covered']}")

print(f"\nText Statistics:")
print(f"   Average words: {overview_stats['text_statistics']['avg_word_count']:.1f}")
print(f"   Median words: {overview_stats['text_statistics']['median_word_count']:.1f}")
print(f"   Word count range: {overview_stats['text_statistics']['min_word_count']} - {overview_stats['text_statistics']['max_word_count']}")

print(f"\nTemporal Distribution:")
print(f"   Market hours: {overview_stats['distribution']['market_hours_pct']:.1f}%")
print(f"   Weekend: {overview_stats['distribution']['weekend_pct']:.1f}%")
print(f"   Weekday: {overview_stats['distribution']['weekday_pct']:.1f}%\n")

# Save overview stats
with open(ANALYSIS_DIR / 'overview_stats.json', 'w') as f:
    json.dump(overview_stats, f, indent=2)

# ============================================================================
# 3. TEMPORAL ANALYSIS
# ============================================================================

print(" Step 3: Analyzing temporal patterns...\n")

# Articles over time
temporal_df = df.groupby(df['date'].dt.to_period('M')).size().reset_index()
temporal_df.columns = ['month', 'article_count']
temporal_df['month'] = temporal_df['month'].astype(str)

# Daily distribution
daily_counts = df.groupby(df['date'].dt.date).size()

# Day of week distribution
dow_dist = df['day_of_week'].value_counts().sort_index()
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Hour distribution
hour_dist = df['hour'].value_counts().sort_index()

print(" Temporal Patterns:")
print(f"   Peak month: {temporal_df.loc[temporal_df['article_count'].idxmax(), 'month']} ({temporal_df['article_count'].max():,} articles)")
print(f"   Average articles/day: {daily_counts.mean():.1f}")
print(f"   Peak day of week: {dow_names[dow_dist.idxmax()]} ({dow_dist.max():,} articles)")
print(f"   Peak hour: {hour_dist.idxmax()}:00 ({hour_dist.max():,} articles)\n")

# Visualization: Articles over time
fig = px.line(
    temporal_df,
    x='month',
    y='article_count',
    title='Article Volume Over Time (Monthly)',
    labels={'month': 'Month', 'article_count': 'Number of Articles'}
)
fig.update_layout(height=500, xaxis_tickangle=-45, hovermode='x unified')
fig.write_html(VIZ_DIR / 'articles_over_time.html')
print(f" Saved: articles_over_time.html")

# Visualization: Day of week distribution
fig2 = px.bar(
    x=[dow_names[i] for i in dow_dist.index],
    y=dow_dist.values,
    title='Article Distribution by Day of Week',
    labels={'x': 'Day of Week', 'y': 'Number of Articles'},
    color=dow_dist.values,
    color_continuous_scale='Blues'
)
fig2.update_layout(height=500, showlegend=False)
fig2.write_html(VIZ_DIR / 'day_of_week_distribution.html')
print(f" Saved: day_of_week_distribution.html")

# Visualization: Hour distribution
fig3 = px.bar(
    x=hour_dist.index,
    y=hour_dist.values,
    title='Article Distribution by Hour of Day',
    labels={'x': 'Hour', 'y': 'Number of Articles'},
    color=hour_dist.values,
    color_continuous_scale='Viridis'
)
fig3.add_vline(x=9, line_dash="dash", line_color="green", annotation_text="Market Open (9 AM)")
fig3.add_vline(x=16, line_dash="dash", line_color="red", annotation_text="Market Close (4 PM)")
fig3.update_layout(height=500, showlegend=False)
fig3.write_html(VIZ_DIR / 'hour_distribution.html')
print(f" Saved: hour_distribution.html\n")

# ============================================================================
# 4. SECTOR AND INDUSTRY ANALYSIS
# ============================================================================

print(" Step 4: Analyzing sector and industry distribution...\n")

# Sector distribution
sector_counts = df['sector'].value_counts()

# Industry distribution (top 20)
industry_counts = df['industry'].value_counts().head(20)

# Articles per company by sector
articles_per_company = df.groupby(['sector', 'ticker']).size().reset_index(name='count')
sector_avg_articles = articles_per_company.groupby('sector')['count'].mean().sort_values(ascending=False)

print(" Sector Analysis:")
print(f"   Top sector: {sector_counts.index[0]} ({sector_counts.iloc[0]:,} articles)")
print(f"   Least covered: {sector_counts.index[-1]} ({sector_counts.iloc[-1]:,} articles)")
print(f"\n   Sector coverage:")
for sector, count in sector_counts.items():
    pct = (count / len(df)) * 100
    print(f"   - {sector}: {count:,} ({pct:.1f}%)")

print(f"\n Top 10 Industries:")
for industry, count in industry_counts.head(10).items():
    print(f"   - {industry}: {count:,} articles\n")

# Visualization: Sector distribution
fig4 = px.pie(
    values=sector_counts.values,
    names=sector_counts.index,
    title='Article Distribution by Sector',
    hole=0.3
)
fig4.update_traces(textposition='inside', textinfo='percent+label')
fig4.write_html(VIZ_DIR / 'sector_distribution.html')
print(f" Saved: sector_distribution.html")

# Visualization: Top industries
fig5 = px.bar(
    x=industry_counts.values,
    y=industry_counts.index,
    orientation='h',
    title='Top 20 Industries by Article Count',
    labels={'x': 'Number of Articles', 'y': 'Industry'},
    color=industry_counts.values,
    color_continuous_scale='Teal'
)
fig5.update_layout(height=700, showlegend=False, yaxis={'categoryorder': 'total ascending'})
fig5.write_html(VIZ_DIR / 'top_industries.html')
print(f" Saved: top_industries.html\n")

# ============================================================================
# 5. COMPANY COVERAGE ANALYSIS
# ============================================================================

print(" Step 5: Analyzing company coverage...\n")

# Articles per company
company_coverage = df.groupby(['ticker', 'company_name', 'sector']).size().reset_index(name='article_count')
company_coverage = company_coverage.sort_values('article_count', ascending=False)

# Coverage statistics
coverage_stats = {
    'mean_articles_per_company': float(company_coverage['article_count'].mean()),
    'median_articles_per_company': float(company_coverage['article_count'].median()),
    'std_articles_per_company': float(company_coverage['article_count'].std()),
    'min_articles': int(company_coverage['article_count'].min()),
    'max_articles': int(company_coverage['article_count'].max()),
    'top_10_companies': company_coverage.head(10)[['ticker', 'company_name', 'article_count']].to_dict('records'),
    'bottom_10_companies': company_coverage.tail(10)[['ticker', 'company_name', 'article_count']].to_dict('records')
}

print(" Company Coverage Statistics:")
print(f"   Mean articles/company: {coverage_stats['mean_articles_per_company']:.1f}")
print(f"   Median articles/company: {coverage_stats['median_articles_per_company']:.1f}")
print(f"   Range: {coverage_stats['min_articles']} - {coverage_stats['max_articles']}")

print(f"\n Top 10 Most Covered Companies:")
for item in coverage_stats['top_10_companies']:
    print(f"   {item['ticker']:6s} - {item['company_name'][:40]:40s}: {item['article_count']:4d} articles")

print(f"\n Bottom 10 Least Covered Companies:")
for item in coverage_stats['bottom_10_companies']:
    print(f"   {item['ticker']:6s} - {item['company_name'][:40]:40s}: {item['article_count']:4d} articles\n")

# Save company coverage
company_coverage.to_csv(ANALYSIS_DIR / 'company_coverage.csv', index=False)
print(f" Saved: company_coverage.csv")

# Visualization: Top 30 companies
top_30 = company_coverage.head(30)
fig6 = px.bar(
    top_30,
    x='article_count',
    y='ticker',
    orientation='h',
    title='Top 30 Companies by Article Count',
    labels={'article_count': 'Number of Articles', 'ticker': 'Ticker'},
    color='article_count',
    color_continuous_scale='RdYlGn',
    hover_data=['company_name', 'sector']
)
fig6.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
fig6.write_html(VIZ_DIR / 'top_companies.html')
print(f" Saved: top_companies.html\n")

# ============================================================================
# 6. TEXT STATISTICS ANALYSIS
# ============================================================================

print(" Step 6: Analyzing text characteristics...\n")

# Word count distribution
word_count_stats = df['word_count'].describe()

# Text length by sector
sector_text_stats = df.groupby('sector').agg({
    'word_count': ['mean', 'median', 'std'],
    'char_count': 'mean',
    'sentence_count': 'mean'
}).round(2)

print(" Text Statistics:")
print(f"   Word count distribution:")
print(f"   - Mean: {word_count_stats['mean']:.1f}")
print(f"   - Median: {word_count_stats['50%']:.1f}")
print(f"   - Std: {word_count_stats['std']:.1f}")
print(f"   - Min: {word_count_stats['min']:.0f}")
print(f"   - Max: {word_count_stats['max']:.0f}")
print(f"   - Q1: {word_count_stats['25%']:.1f}")
print(f"   - Q3: {word_count_stats['75%']:.1f}")

print(f"\n Average Word Count by Sector:")
for sector in sector_text_stats.index:
    avg_words = sector_text_stats.loc[sector, ('word_count', 'mean')]
    print(f"   {sector}: {avg_words:.1f} words\n")

# Save text statistics
text_stats_df = pd.DataFrame({
    'metric': ['mean', 'median', 'std', 'min', 'max', 'q1', 'q3'],
    'word_count': [
        word_count_stats['mean'], word_count_stats['50%'], word_count_stats['std'],
        word_count_stats['min'], word_count_stats['max'],
        word_count_stats['25%'], word_count_stats['75%']
    ],
    'char_count': [
        df['char_count'].mean(), df['char_count'].median(), df['char_count'].std(),
        df['char_count'].min(), df['char_count'].max(),
        df['char_count'].quantile(0.25), df['char_count'].quantile(0.75)
    ],
    'sentence_count': [
        df['sentence_count'].mean(), df['sentence_count'].median(), df['sentence_count'].std(),
        df['sentence_count'].min(), df['sentence_count'].max(),
        df['sentence_count'].quantile(0.25), df['sentence_count'].quantile(0.75)
    ]
})
text_stats_df.to_csv(ANALYSIS_DIR / 'text_statistics.csv', index=False)
print(f" Saved: text_statistics.csv")

# Visualization: Word count distribution
fig7 = px.histogram(
    df,
    x='word_count',
    nbins=50,
    title='Word Count Distribution',
    labels={'word_count': 'Word Count', 'count': 'Frequency'},
    color_discrete_sequence=['steelblue']
)
fig7.add_vline(x=df['word_count'].mean(), line_dash="dash", line_color="red", 
               annotation_text=f"Mean: {df['word_count'].mean():.1f}")
fig7.update_layout(height=500)
fig7.write_html(VIZ_DIR / 'word_count_distribution.html')
print(f" Saved: word_count_distribution.html\n")

# ============================================================================
# 7. NEWS SOURCE ANALYSIS
# ============================================================================

print(" Step 7: Analyzing news sources...\n")

# Source distribution
source_counts = df['source'].value_counts()
top_sources = source_counts.head(20)

# Sources by sector
sources_by_sector = df.groupby(['sector', 'source']).size().reset_index(name='count')

print(" News Source Analysis:")
print(f"   Total unique sources: {len(source_counts)}")
print(f"   Top source: {source_counts.index[0]} ({source_counts.iloc[0]:,} articles)")

print(f"\n   Top 15 News Sources:")
for idx, (source, count) in enumerate(top_sources.head(15).items(), 1):
    pct = (count / len(df)) * 100
    print(f"   {idx:2d}. {source[:50]:50s}: {count:5,} ({pct:4.1f}%)\n")

# Visualization: Top sources
fig8 = px.bar(
    x=top_sources.values,
    y=top_sources.index,
    orientation='h',
    title='Top 20 News Sources by Article Count',
    labels={'x': 'Number of Articles', 'y': 'Source'},
    color=top_sources.values,
    color_continuous_scale='Purp'
)
fig8.update_layout(height=700, showlegend=False, yaxis={'categoryorder': 'total ascending'})
fig8.write_html(VIZ_DIR / 'top_news_sources.html')
print(f" Saved: top_news_sources.html\n")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================

print(" Step 8: Analyzing feature correlations...\n")

# Select numeric columns for correlation
numeric_cols = ['word_count', 'char_count', 'sentence_count', 'hour', 'day_of_week', 
                'is_weekend', 'is_market_hours']
corr_matrix = df[numeric_cols].corr()

print(" Feature Correlations:")
print(corr_matrix.round(3))
print()

# Visualization: Correlation heatmap
fig9 = px.imshow(
    corr_matrix,
    text_auto='.2f',
    title='Feature Correlation Matrix',
    color_continuous_scale='RdBu_r',
    aspect='auto'
)
fig9.update_layout(height=600)
fig9.write_html(VIZ_DIR / 'correlation_matrix.html')
print(f" Saved: correlation_matrix.html\n")

# ============================================================================
# 9. GENERATE COMPREHENSIVE REPORT
# ============================================================================

print(" Step 9: Generating comprehensive EDA report...\n")

eda_report = {
    'overview': overview_stats,
    'temporal_analysis': {
        'articles_per_month': temporal_df.to_dict('records'),
        'avg_articles_per_day': float(daily_counts.mean()),
        'peak_month': temporal_df.loc[temporal_df['article_count'].idxmax(), 'month'],
        'peak_day_of_week': dow_names[dow_dist.idxmax()],
        'peak_hour': int(hour_dist.idxmax())
    },
    'sector_analysis': {
        'distribution': sector_counts.to_dict(),
        'avg_articles_per_company_by_sector': sector_avg_articles.to_dict()
    },
    'company_coverage': coverage_stats,
    'text_analysis': {
        'word_count_stats': word_count_stats.to_dict(),
        'avg_by_sector': sector_text_stats.to_dict()
    },
    'source_analysis': {
        'total_sources': int(len(source_counts)),
        'top_20_sources': top_sources.to_dict()
    },
    'correlation_matrix': corr_matrix.to_dict()
}

# Save comprehensive report
with open(ANALYSIS_DIR / 'full_eda_results.json', 'w') as f:
    json.dump(eda_report, f, indent=2, default=str)

print(f" Saved: full_eda_results.json")

# ============================================================================
# 10. GENERATE MARKDOWN REPORT
# ============================================================================

markdown_report = f"""# Exploratory Data Analysis Report
## Financial News Dataset

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Dataset Overview

- **Total Articles:** {overview_stats['dataset_size']['total_articles']:,}
- **Companies Covered:** {overview_stats['dataset_size']['unique_companies']:,}
- **Sectors:** {overview_stats['dataset_size']['unique_sectors']}
- **Industries:** {overview_stats['dataset_size']['unique_industries']}
- **News Sources:** {overview_stats['dataset_size']['unique_sources']}

### Temporal Coverage
- **Date Range:** {overview_stats['temporal_coverage']['start_date']} to {overview_stats['temporal_coverage']['end_date']}
- **Days Covered:** {overview_stats['temporal_coverage']['days_covered']:,}
- **Years:** {overview_stats['temporal_coverage']['years_covered']}
- **Months:** {overview_stats['temporal_coverage']['months_covered']}

### Text Statistics
- **Average Word Count:** {overview_stats['text_statistics']['avg_word_count']:.1f}
- **Median Word Count:** {overview_stats['text_statistics']['median_word_count']:.1f}
- **Word Count Range:** {overview_stats['text_statistics']['min_word_count']} - {overview_stats['text_statistics']['max_word_count']}

---

## Sector Distribution

"""

for sector, count in sector_counts.items():
    pct = (count / len(df)) * 100
    markdown_report += f"- **{sector}:** {count:,} articles ({pct:.1f}%)\n"

markdown_report += f"""
---

## Top 10 Companies by Coverage

"""

for item in coverage_stats['top_10_companies']:
    markdown_report += f"{item['ticker']} - {item['company_name']}: {item['article_count']} articles\n"

markdown_report += f"""
---

## Temporal Patterns

- **Peak Month:** {eda_report['temporal_analysis']['peak_month']}
- **Average Articles/Day:** {eda_report['temporal_analysis']['avg_articles_per_day']:.1f}
- **Peak Day of Week:** {eda_report['temporal_analysis']['peak_day_of_week']}
- **Peak Hour:** {eda_report['temporal_analysis']['peak_hour']}:00

---

## Key Insights

1. **Coverage is comprehensive** with {overview_stats['dataset_size']['total_articles']:,} articles across {overview_stats['dataset_size']['unique_companies']:,} companies
2. **Text quality is high** with average {overview_stats['text_statistics']['avg_word_count']:.1f} words per article
3. **Temporal distribution** shows {overview_stats['distribution']['market_hours_pct']:.1f}% of articles published during market hours
4. **Sector diversity** with coverage across {overview_stats['dataset_size']['unique_sectors']} major sectors
5. **Source variety** with {eda_report['source_analysis']['total_sources']} unique news sources

---

*Report generated automatically by EDA script*
"""

with open(ANALYSIS_DIR / 'financial_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_report)

print(f" Saved: financial_report.md\n")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("="*80)
print("EXPLORATORY DATA ANALYSIS COMPLETE")
print("="*80)

print(f"\n Analysis Summary:")
print(f"   Analyzed {len(df):,} articles")
print(f"   Generated {len(list(VIZ_DIR.glob('*.html')))} visualizations")
print(f"   Created {len(list(ANALYSIS_DIR.glob('*')))} analysis files")

print(f"\n Output Files:")
print(f"   Analysis:")
for file in sorted(ANALYSIS_DIR.glob('*')):
    print(f"   - {file.name}")

print(f"\n   Visualizations:")
for file in sorted(VIZ_DIR.glob('*.html')):
    print(f"   - {file.name}")

print(f"\n Ready for Step 3: Sentiment Analysis")
print("="*80)
