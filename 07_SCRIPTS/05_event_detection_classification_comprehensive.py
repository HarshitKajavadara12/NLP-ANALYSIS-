"""
Event Detection and Classification - Comprehensive
==================================================

Detects and classifies 12 financial event types:
1. Earnings Reports       7. Legal Issues
2. Mergers & Acquisitions 8. Partnerships
3. Product Launches       9. Disruptions
4. Regulatory Changes     10. Dividend Announcements
5. Executive Changes      11. Restructuring
6. Analyst Ratings        12. Economic Indicators

Uses rule-based pattern matching with financial lexicon
Target: F1-score >0.82
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
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
EVENTS_DIR = DATA_DIR / 'events'
CONFIG_DIR = BASE_DIR / '06_CONFIG'
VIZ_DIR = DATA_DIR / 'visualizations'

# Create directories
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EVENT DETECTION & CLASSIFICATION - COMPREHENSIVE")
print("="*80)
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print(" Step 1: Loading dataset with entities...")

df = pd.read_csv(PROCESSED_DIR / 'dataset_with_entities.csv')
df['date'] = pd.to_datetime(df['date'])

print(f" Loaded {len(df):,} articles")
print(f"   Companies: {df['ticker'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}\n")

# ============================================================================
# 2. LOAD EVENT CONFIGURATION
# ============================================================================

print(" Step 2: Loading event configuration...")

# Load event patterns from config
with open(CONFIG_DIR / 'financial_events.json', 'r') as f:
    event_config = json.load(f)

event_patterns = event_config['event_types']

print(f" Loaded {len(event_patterns)} event types:")
for event in event_patterns:
    print(f"   {event['id']}. {event['name']}")
print()

# ============================================================================
# 3. EVENT DETECTOR CLASS
# ============================================================================

print(" Step 3: Initializing Event Detector...")

class FinancialEventDetector:
    """Comprehensive financial event detector"""
    
    def __init__(self, event_patterns):
        self.event_patterns = event_patterns
        self.event_map = {event['name']: event for event in event_patterns}
    
    def detect_events(self, text):
        """Detect all events in text"""
        text_lower = text.lower()
        detected_events = {}
        
        for event in self.event_patterns:
            event_name = event['name']
            score = 0
            
            # Check regex patterns (higher weight)
            for pattern in event.get('patterns', []):
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 3
            
            # Check keywords (lower weight)
            for keyword in event.get('keywords', []):
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                detected_events[event_name] = {
                    'score': score,
                    'id': event['id'],
                    'impact': event.get('market_impact', 'MEDIUM')
                }
        
        return detected_events
    
    def get_primary_event(self, events):
        """Get primary event (highest score)"""
        if not events:
            return None
        
        return max(events.items(), key=lambda x: x[1]['score'])[0]
    
    def calculate_event_impact_score(self, events, sentiment, num_entities):
        """Calculate overall event impact score"""
        if not events:
            return 0.0
        
        # Base impact from events
        impact_weights = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}
        base_impact = sum(
            impact_weights.get(evt['impact'], 0.5) * evt['score'] 
            for evt in events.values()
        )
        
        # Normalize base impact
        base_impact = min(base_impact / 10, 1.0)
        
        # Combine with sentiment magnitude and entity richness
        sentiment_magnitude = abs(sentiment)
        entity_factor = min(num_entities / 20, 1.0)
        
        impact_score = (
            base_impact * 0.5 +
            sentiment_magnitude * 0.3 +
            entity_factor * 0.2
        )
        
        return min(impact_score, 1.0)

detector = FinancialEventDetector(event_patterns)
print(f" Event detector initialized\n")

# ============================================================================
# 4. EVENT DETECTION
# ============================================================================

print(" Step 4: Detecting events in all articles...")
print(f"   Processing {len(df):,} articles...\n")

# Detect events
detected_events_list = []
for text in tqdm(df['text'], desc="Event detection"):
    events = detector.detect_events(text)
    detected_events_list.append(events)

df['detected_events'] = detected_events_list

# Extract event information
df['num_events'] = df['detected_events'].apply(len)
df['primary_event'] = df['detected_events'].apply(detector.get_primary_event)
df['event_impact_score'] = df.apply(
    lambda row: detector.calculate_event_impact_score(
        row['detected_events'],
        row['finbert_sentiment'],
        row['num_entities']
    ),
    axis=1
)

print(f"\n Event detection complete")
print(f"   Articles with events: {(df['num_events'] > 0).sum():,} ({(df['num_events'] > 0).mean()*100:.1f}%)")
print(f"   Avg events per article: {df['num_events'].mean():.2f}")
print(f"   Max events in one article: {df['num_events'].max()}\n")

# ============================================================================
# 5. CREATE EVENT INDICATORS
# ============================================================================

print(" Step 5: Creating event type indicators...")

# Create binary indicators for each event type
event_type_names = [event['name'] for event in event_patterns]

for event_name in event_type_names:
    col_name = f"has_{event_name.lower().replace(' ', '_').replace('&', 'and')}"
    df[col_name] = df['detected_events'].apply(lambda x: 1 if event_name in x else 0)

# High impact event indicator
df['has_high_impact_event'] = df['detected_events'].apply(
    lambda x: 1 if any(evt['impact'] == 'HIGH' for evt in x.values()) else 0
)

print(f" Event indicators created")
print(f"   Event type columns: {len(event_type_names)}")
print(f"   High impact events: {df['has_high_impact_event'].sum():,} ({df['has_high_impact_event'].mean()*100:.1f}%)\n")

# ============================================================================
# 6. EVENT DISTRIBUTION ANALYSIS
# ============================================================================

print(" Step 6: Analyzing event distribution...\n")

# Count each event type
event_counts = {}
for event_name in event_type_names:
    col_name = f"has_{event_name.lower().replace(' ', '_').replace('&', 'and')}"
    count = df[col_name].sum()
    event_counts[event_name] = int(count)

event_dist = pd.DataFrame.from_dict(event_counts, orient='index', columns=['count'])
event_dist = event_dist.sort_values('count', ascending=False)

print(f" Event Distribution:")
for event_name, count in event_dist.iterrows():
    pct = (count['count'] / len(df)) * 100
    print(f"   {event_name:25s}: {count['count']:6,} ({pct:5.2f}%)")
print()

# Save event distribution
event_dist.to_csv(EVENTS_DIR / 'event_distribution.csv')
print(f" Saved: event_distribution.csv\n")

# ============================================================================
# 7. PRIMARY EVENT ANALYSIS
# ============================================================================

print(" Step 7: Analyzing primary events...")

# Primary event distribution
primary_event_dist = df[df['primary_event'].notna()]['primary_event'].value_counts()

print(f" Primary Event Distribution:")
print(f"   (Most dominant event per article)\n")
for event, count in primary_event_dist.items():
    pct = (count / len(df)) * 100
    print(f"   {event:25s}: {count:6,} ({pct:5.2f}%)")
print()

# ============================================================================
# 8. EVENT-SENTIMENT CORRELATION
# ============================================================================

print(" Step 8: Analyzing event-sentiment correlation...")

# Average sentiment by event type
event_sentiment = {}
for event_name in event_type_names:
    col_name = f"has_{event_name.lower().replace(' ', '_').replace('&', 'and')}"
    subset = df[df[col_name] == 1]
    if len(subset) > 0:
        avg_sentiment = subset['finbert_sentiment'].mean()
        event_sentiment[event_name] = float(avg_sentiment)
    else:
        event_sentiment[event_name] = 0.0

event_sentiment_df = pd.DataFrame.from_dict(
    event_sentiment, orient='index', columns=['avg_sentiment']
).sort_values('avg_sentiment', ascending=False)

print(f" Average Sentiment by Event Type:\n")
print(event_sentiment_df)
print()

# Save
event_sentiment_df.to_csv(EVENTS_DIR / 'event_sentiment_correlation.csv')
print(f" Saved: event_sentiment_correlation.csv\n")

# ============================================================================
# 9. EVENT CO-OCCURRENCE ANALYSIS
# ============================================================================

print(" Step 9: Analyzing event co-occurrence...")

# Create co-occurrence matrix
cooccurrence = np.zeros((len(event_type_names), len(event_type_names)))

for idx, row in df.iterrows():
    events = row['detected_events']
    event_list = list(events.keys())
    
    for i, event1 in enumerate(event_type_names):
        for j, event2 in enumerate(event_type_names):
            if event1 in event_list and event2 in event_list and i != j:
                cooccurrence[i, j] += 1

cooccurrence_df = pd.DataFrame(
    cooccurrence,
    index=event_type_names,
    columns=event_type_names
)

print(f" Event co-occurrence matrix calculated")
print(f"   Most common pair:")
max_val = 0
max_pair = None
for i in range(len(event_type_names)):
    for j in range(i+1, len(event_type_names)):
        if cooccurrence[i, j] > max_val:
            max_val = cooccurrence[i, j]
            max_pair = (event_type_names[i], event_type_names[j])

if max_pair:
    print(f"   {max_pair[0]} + {max_pair[1]}: {int(max_val)} times\n")

# Save
cooccurrence_df.to_csv(EVENTS_DIR / 'event_cooccurrence_matrix.csv')
print(f" Saved: event_cooccurrence_matrix.csv\n")

# ============================================================================
# 10. TEMPORAL EVENT ANALYSIS
# ============================================================================

print(" Step 10: Analyzing temporal event patterns...")

# Events over time
df['year_month'] = df['date'].dt.to_period('M').astype(str)
temporal_events = df.groupby('year_month').agg({
    'num_events': ['mean', 'sum'],
    'has_high_impact_event': 'sum',
    'ticker': 'count'
}).reset_index()

temporal_events.columns = ['month', 'avg_events', 'total_events', 'high_impact_count', 'article_count']

print(f" Temporal event patterns analyzed")
print(f"   Peak month: {temporal_events.loc[temporal_events['total_events'].idxmax(), 'month']}")
print(f"   Max events: {temporal_events['total_events'].max():.0f}\n")

# Save
temporal_events.to_csv(EVENTS_DIR / 'temporal_event_patterns.csv', index=False)
print(f" Saved: temporal_event_patterns.csv\n")

# ============================================================================
# 11. COMPANY-EVENT PROFILES
# ============================================================================

print(" Step 11: Creating company-event profiles...")

# Most common event per company
company_events = []
for ticker in df['ticker'].unique():
    ticker_df = df[df['ticker'] == ticker]
    company_name = ticker_df['company_name'].iloc[0]
    sector = ticker_df['sector'].iloc[0]
    
    # Count events for this company
    event_profile = {}
    for event_name in event_type_names:
        col_name = f"has_{event_name.lower().replace(' ', '_').replace('&', 'and')}"
        count = ticker_df[col_name].sum()
        event_profile[event_name] = int(count)
    
    # Primary event
    if event_profile:
        primary = max(event_profile.items(), key=lambda x: x[1])
        
        company_events.append({
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'total_events': ticker_df['num_events'].sum(),
            'avg_events_per_article': ticker_df['num_events'].mean(),
            'primary_event': primary[0],
            'primary_event_count': primary[1],
            'high_impact_events': ticker_df['has_high_impact_event'].sum()
        })

company_event_profiles = pd.DataFrame(company_events)
company_event_profiles = company_event_profiles.sort_values('total_events', ascending=False)

print(f" Company-event profiles created")
print(f"   Companies: {len(company_event_profiles)}")
print(f"\n   Top 10 companies by total events:")
print(company_event_profiles.head(10)[['ticker', 'company_name', 'total_events', 'primary_event']].to_string(index=False))
print()

# Save
company_event_profiles.to_csv(EVENTS_DIR / 'company_event_profiles.csv', index=False)
print(f" Saved: company_event_profiles.csv\n")

# ============================================================================
# 12. VISUALIZATIONS
# ============================================================================

print(" Step 12: Creating visualizations...\n")

# 1. Event distribution
fig1 = px.bar(
    event_dist.reset_index(),
    x='index',
    y='count',
    title='Financial Event Type Distribution',
    labels={'index': 'Event Type', 'count': 'Number of Occurrences'},
    color='count',
    color_continuous_scale='Viridis'
)
fig1.update_layout(height=600, xaxis_tickangle=-45, showlegend=False)
fig1.write_html(VIZ_DIR / 'event_distribution.html')
print(f" Saved: event_distribution.html")

# 2. Event-sentiment correlation
fig2 = px.bar(
    event_sentiment_df.reset_index(),
    x='index',
    y='avg_sentiment',
    title='Average Sentiment by Event Type',
    labels={'index': 'Event Type', 'avg_sentiment': 'Average Sentiment'},
    color='avg_sentiment',
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(height=600, xaxis_tickangle=-45)
fig2.write_html(VIZ_DIR / 'event_sentiment_correlation.html')
print(f" Saved: event_sentiment_correlation.html")

# 3. Events over time
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=temporal_events['month'],
    y=temporal_events['total_events'],
    mode='lines+markers',
    name='Total Events',
    line=dict(color='steelblue', width=2)
))
fig3.update_layout(
    title='Event Detection Over Time',
    xaxis_title='Month',
    yaxis_title='Number of Events Detected',
    height=500,
    xaxis_tickangle=-45,
    hovermode='x unified'
)
fig3.write_html(VIZ_DIR / 'events_over_time.html')
print(f" Saved: events_over_time.html")

# 4. Event co-occurrence heatmap
fig4 = px.imshow(
    cooccurrence_df,
    text_auto='.0f',
    title='Event Co-occurrence Matrix',
    color_continuous_scale='YlOrRd',
    aspect='auto'
)
fig4.update_layout(height=700, xaxis_tickangle=-45)
fig4.write_html(VIZ_DIR / 'event_cooccurrence_heatmap.html')
print(f" Saved: event_cooccurrence_heatmap.html\n")

# ============================================================================
# 13. SAVE ENHANCED DATASET
# ============================================================================

print(" Step 13: Saving enhanced dataset with events...")

# Convert detected_events dict to JSON string
df['detected_events_json'] = df['detected_events'].apply(json.dumps)

# Select columns
event_columns = [
    'ticker', 'company_name', 'sector', 'industry', 'date',
    'title', 'text', 'source',
    'finbert_sentiment', 'ensemble_sentiment',
    'num_entities', 'num_orgs', 'num_persons',
    'num_events', 'primary_event', 'event_impact_score',
    'has_high_impact_event',
    'detected_events_json',
    'word_count', 'year', 'month', 'is_weekend', 'is_market_hours'
]

# Add all event type indicators
for event_name in event_type_names:
    col_name = f"has_{event_name.lower().replace(' ', '_').replace('&', 'and')}"
    event_columns.append(col_name)

df_events = df[event_columns].copy()

# Save
output_file = PROCESSED_DIR / 'dataset_with_events.csv'
df_events.to_csv(output_file, index=False)

print(f" Enhanced dataset saved to {output_file}")
print(f"   Rows: {len(df_events):,}")
print(f"   Columns: {len(df_events.columns)}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

# ============================================================================
# 14. GENERATE SUMMARY
# ============================================================================

print(" Step 14: Generating event detection summary...\n")

summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'total_articles': int(len(df)),
        'articles_with_events': int((df['num_events'] > 0).sum()),
        'articles_with_events_pct': float((df['num_events'] > 0).mean() * 100)
    },
    'event_detection': {
        'total_events_detected': int(df['num_events'].sum()),
        'avg_events_per_article': float(df['num_events'].mean()),
        'max_events_per_article': int(df['num_events'].max()),
        'high_impact_events': int(df['has_high_impact_event'].sum())
    },
    'event_distribution': event_counts,
    'primary_events': primary_event_dist.to_dict(),
    'event_sentiment_correlation': event_sentiment_df.to_dict()['avg_sentiment'],
    'temporal_patterns': {
        'peak_month': temporal_events.loc[temporal_events['total_events'].idxmax(), 'month'],
        'max_events_in_month': int(temporal_events['total_events'].max())
    },
    'top_companies_by_events': company_event_profiles.head(10).to_dict('records')
}

# Save
summary_file = EVENTS_DIR / 'event_detection_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f" Saved: event_detection_summary.json")

# ============================================================================
# 15. FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("EVENT DETECTION & CLASSIFICATION COMPLETE")
print("="*80)

print(f"\n Event Detection Summary:")
print(f"   Articles analyzed: {summary['dataset']['total_articles']:,}")
print(f"   Articles with events: {summary['dataset']['articles_with_events']:,} ({summary['dataset']['articles_with_events_pct']:.1f}%)")
print(f"   Total events detected: {summary['event_detection']['total_events_detected']:,}")
print(f"   Avg events/article: {summary['event_detection']['avg_events_per_article']:.2f}")

print(f"\n Event Type Distribution:")
top_3_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:3]
for event, count in top_3_events:
    pct = (count / len(df)) * 100
    print(f"   {event}: {count:,} ({pct:.1f}%)")

print(f"\n High Impact Events: {summary['event_detection']['high_impact_events']:,}")

print(f"\n Output Files:")
print(f"   1. {output_file}")
print(f"   2. {EVENTS_DIR / 'event_distribution.csv'}")
print(f"   3. {EVENTS_DIR / 'event_sentiment_correlation.csv'}")
print(f"   4. {EVENTS_DIR / 'event_cooccurrence_matrix.csv'}")
print(f"   5. {EVENTS_DIR / 'company_event_profiles.csv'}")
print(f"   6. {summary_file}")

print(f"\n Ready for Step 6: Topic Modeling & Classification")
print("="*80)
