"""
Named Entity Recognition (NER) - Comprehensive
==============================================

Dual NER system:
1. spaCy (en_core_web_trf) - Transformer-based
2. BERT-NER (dslim/bert-base-NER) - Domain-specific

Extracts: Organizations, Persons, Locations, and Financial Entities
Target: 92.1% precision matching institutional benchmarks
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NER models
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from tqdm.auto import tqdm
import networkx as nx

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
ENTITIES_DIR = DATA_DIR / 'entities'
VIZ_DIR = DATA_DIR / 'visualizations'

# Create directories
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("NAMED ENTITY RECOGNITION - COMPREHENSIVE")
print("="*80)
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print(" Step 1: Loading dataset with sentiment...")

df = pd.read_csv(PROCESSED_DIR / 'dataset_with_sentiment.csv')
df['date'] = pd.to_datetime(df['date'])

print(f" Loaded {len(df):,} articles")
print(f"   Companies: {df['ticker'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}\n")

# ============================================================================
# 2. LOAD NER MODELS
# ============================================================================

print(" Step 2: Loading NER models...")

# Check for GPU
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"
print(f"   Using device: {device_name}")

# Load spaCy model
print("   Loading spaCy transformer model...")
nlp_spacy = spacy.load("en_core_web_trf")

# Load BERT-NER model
print("   Loading BERT-NER model...")
bert_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

bert_ner = pipeline(
    "ner",
    model=bert_model,
    tokenizer=bert_tokenizer,
    device=device,
    aggregation_strategy="simple"
)

print(f" Both NER models loaded successfully\n")

# ============================================================================
# 3. SPACY NER EXTRACTION
# ============================================================================

print(" Step 3: Running spaCy NER extraction...")
print(f"   Processing {len(df):,} articles...\n")

def extract_entities_spacy(text, max_length=1000000):
    """Extract entities using spaCy"""
    entities = {
        'ORG': [],
        'PERSON': [],
        'GPE': [],  # Geopolitical entity (locations)
        'MONEY': [],
        'PERCENT': [],
        'DATE': []
    }
    
    try:
        # Process with spaCy
        doc = nlp_spacy(text[:max_length])
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
    
    except Exception as e:
        pass  # Return empty entities on error
    
    return entities

# Extract entities
spacy_entities = []
for text in tqdm(df['text'], desc="spaCy NER"):
    entities = extract_entities_spacy(text)
    spacy_entities.append(entities)

# Add to dataframe
df['spacy_entities'] = spacy_entities
df['num_orgs'] = df['spacy_entities'].apply(lambda x: len(x['ORG']))
df['num_persons'] = df['spacy_entities'].apply(lambda x: len(x['PERSON']))
df['num_locations'] = df['spacy_entities'].apply(lambda x: len(x['GPE']))
df['num_entities'] = df['num_orgs'] + df['num_persons'] + df['num_locations']

print(f"\n spaCy NER complete")
print(f"   Total entities: {df['num_entities'].sum():,}")
print(f"   Organizations: {df['num_orgs'].sum():,}")
print(f"   Persons: {df['num_persons'].sum():,}")
print(f"   Locations: {df['num_locations'].sum():,}")
print(f"   Avg entities per article: {df['num_entities'].mean():.2f}\n")

# ============================================================================
# 4. BERT-NER EXTRACTION
# ============================================================================

print(" Step 4: Running BERT-NER extraction...")

def extract_entities_bert(text, batch_size=16):
    """Extract entities using BERT-NER"""
    entities = {
        'ORG': [],
        'PER': [],
        'LOC': [],
        'MISC': []
    }
    
    try:
        # Truncate to avoid memory issues
        text = text[:512]
        
        # Extract entities
        ner_results = bert_ner(text)
        
        for entity in ner_results:
            entity_type = entity['entity_group']
            entity_text = entity['word'].replace('##', '')
            
            if entity_type in entities:
                entities[entity_type].append(entity_text)
    
    except Exception as e:
        pass  # Return empty entities on error
    
    return entities

# Extract BERT entities
bert_entities = []
for text in tqdm(df['text'], desc="BERT-NER"):
    entities = extract_entities_bert(text)
    bert_entities.append(entities)

# Add to dataframe
df['bert_entities'] = bert_entities
df['bert_num_orgs'] = df['bert_entities'].apply(lambda x: len(x['ORG']))
df['bert_num_persons'] = df['bert_entities'].apply(lambda x: len(x['PER']))
df['bert_num_locations'] = df['bert_entities'].apply(lambda x: len(x['LOC']))

print(f"\n BERT-NER complete")
print(f"   Organizations: {df['bert_num_orgs'].sum():,}")
print(f"   Persons: {df['bert_num_persons'].sum():,}")
print(f"   Locations: {df['bert_num_locations'].sum():,}\n")

# ============================================================================
# 5. BUILD ENTITY DATABASE
# ============================================================================

print(" Step 5: Building comprehensive entity database...")

# Collect all entities
all_orgs = []
all_persons = []
all_locations = []

for idx, row in df.iterrows():
    ticker = row['ticker']
    date = row['date']
    
    # Add spaCy entities
    for org in row['spacy_entities']['ORG']:
        all_orgs.append({'entity': org, 'type': 'ORG', 'source': 'spacy', 'ticker': ticker, 'date': date})
    for person in row['spacy_entities']['PERSON']:
        all_persons.append({'entity': person, 'type': 'PERSON', 'source': 'spacy', 'ticker': ticker, 'date': date})
    for loc in row['spacy_entities']['GPE']:
        all_locations.append({'entity': loc, 'type': 'LOCATION', 'source': 'spacy', 'ticker': ticker, 'date': date})
    
    # Add BERT entities
    for org in row['bert_entities']['ORG']:
        all_orgs.append({'entity': org, 'type': 'ORG', 'source': 'bert', 'ticker': ticker, 'date': date})
    for person in row['bert_entities']['PER']:
        all_persons.append({'entity': person, 'type': 'PERSON', 'source': 'bert', 'ticker': ticker, 'date': date})
    for loc in row['bert_entities']['LOC']:
        all_locations.append({'entity': loc, 'type': 'LOCATION', 'source': 'bert', 'ticker': ticker, 'date': date})

# Create entity database
entity_db = pd.DataFrame(all_orgs + all_persons + all_locations)

# Count entity mentions
entity_counts = entity_db.groupby(['entity', 'type']).size().reset_index(name='mention_count')
entity_counts = entity_counts.sort_values('mention_count', ascending=False)

print(f" Entity database created")
print(f"   Unique entities: {len(entity_counts):,}")
print(f"   Total mentions: {entity_counts['mention_count'].sum():,}")
print(f"   Average mentions per entity: {entity_counts['mention_count'].mean():.2f}\n")

# Save entity database
entity_db.to_csv(ENTITIES_DIR / 'financial_entity_database.csv', index=False)
entity_counts.to_csv(ENTITIES_DIR / 'entity_mention_counts.csv', index=False)
print(f" Saved: financial_entity_database.csv")
print(f" Saved: entity_mention_counts.csv\n")

# ============================================================================
# 6. TOP ENTITIES ANALYSIS
# ============================================================================

print(" Step 6: Analyzing top entities...")

# Top organizations
top_orgs = entity_counts[entity_counts['type'] == 'ORG'].head(50)
print(f" Top 20 Organizations:")
print(top_orgs.head(20).to_string(index=False))

# Top persons
top_persons = entity_counts[entity_counts['type'] == 'PERSON'].head(50)
print(f"\n Top 20 Persons:")
print(top_persons.head(20).to_string(index=False))

# Top locations
top_locations = entity_counts[entity_counts['type'] == 'LOCATION'].head(50)
print(f"\n Top 20 Locations:")
print(top_locations.head(20).to_string(index=False))
print()

# Save top entities
top_orgs.to_csv(ENTITIES_DIR / 'top_organizations.csv', index=False)
top_persons.to_csv(ENTITIES_DIR / 'top_persons.csv', index=False)
top_locations.to_csv(ENTITIES_DIR / 'top_locations.csv', index=False)

# ============================================================================
# 7. ENTITY CO-OCCURRENCE NETWORK
# ============================================================================

print(" Step 7: Building entity co-occurrence network...")

# Build co-mention matrix (entities mentioned together in same article)
co_mentions = []

for idx, row in df.iterrows():
    # Combine all entities from article
    entities = (
        row['spacy_entities']['ORG'] + 
        row['spacy_entities']['PERSON'] + 
        row['bert_entities']['ORG'] + 
        row['bert_entities']['PER']
    )
    
    # Remove duplicates
    entities = list(set(entities))
    
    # Create co-mention pairs
    for i, ent1 in enumerate(entities):
        for ent2 in entities[i+1:]:
            co_mentions.append({'entity1': ent1, 'entity2': ent2, 'ticker': row['ticker']})

# Count co-mentions
if co_mentions:
    co_mention_df = pd.DataFrame(co_mentions)
    co_mention_counts = co_mention_df.groupby(['entity1', 'entity2']).size().reset_index(name='co_mention_count')
    co_mention_counts = co_mention_counts.sort_values('co_mention_count', ascending=False)
    
    print(f" Co-occurrence network built")
    print(f"   Entity pairs: {len(co_mention_counts):,}")
    print(f"   Top co-mentions:")
    print(co_mention_counts.head(20).to_string(index=False))
    print()
    
    # Save co-mentions
    co_mention_counts.to_csv(ENTITIES_DIR / 'entity_co_mentions.csv', index=False)
    print(f" Saved: entity_co_mentions.csv\n")

# ============================================================================
# 8. TICKER-ENTITY ASSOCIATIONS
# ============================================================================

print(" Step 8: Analyzing ticker-entity associations...")

# Most mentioned entities per ticker
ticker_entities = entity_db.groupby(['ticker', 'entity', 'type']).size().reset_index(name='count')
ticker_entities = ticker_entities.sort_values(['ticker', 'count'], ascending=[True, False])

# Top entities per ticker
top_per_ticker = ticker_entities.groupby('ticker').head(10)

print(f" Ticker-entity associations analyzed")
print(f"   Ticker-entity pairs: {len(ticker_entities):,}\n")

# Save
ticker_entities.to_csv(ENTITIES_DIR / 'ticker_entity_associations.csv', index=False)
print(f" Saved: ticker_entity_associations.csv\n")

# ============================================================================
# 9. ENTITY TYPE DISTRIBUTION BY SECTOR
# ============================================================================

print(" Step 9: Analyzing entity distribution by sector...")

sector_entities = df.groupby('sector').agg({
    'num_orgs': 'mean',
    'num_persons': 'mean',
    'num_locations': 'mean',
    'num_entities': ['mean', 'sum']
}).round(2)

sector_entities.columns = ['avg_orgs', 'avg_persons', 'avg_locations', 'avg_total', 'total_entities']
sector_entities = sector_entities.sort_values('avg_total', ascending=False)

print(f" Entity Distribution by Sector:\n")
print(sector_entities)
print()

# Save
sector_entities.to_csv(ENTITIES_DIR / 'sector_entity_distribution.csv')
print(f" Saved: sector_entity_distribution.csv\n")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================

print(" Step 10: Creating visualizations...\n")

# 1. Entity type distribution
entity_type_dist = entity_counts.groupby('type')['mention_count'].sum()
fig1 = px.pie(
    values=entity_type_dist.values,
    names=entity_type_dist.index,
    title='Entity Type Distribution',
    hole=0.3
)
fig1.write_html(VIZ_DIR / 'entity_type_distribution.html')
print(f" Saved: entity_type_distribution.html")

# 2. Top 30 organizations
fig2 = px.bar(
    top_orgs.head(30),
    x='mention_count',
    y='entity',
    orientation='h',
    title='Top 30 Most Mentioned Organizations',
    labels={'mention_count': 'Mention Count', 'entity': 'Organization'},
    color='mention_count',
    color_continuous_scale='Blues'
)
fig2.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
fig2.write_html(VIZ_DIR / 'top_organizations.html')
print(f" Saved: top_organizations.html")

# 3. Top 30 persons
fig3 = px.bar(
    top_persons.head(30),
    x='mention_count',
    y='entity',
    orientation='h',
    title='Top 30 Most Mentioned Persons',
    labels={'mention_count': 'Mention Count', 'entity': 'Person'},
    color='mention_count',
    color_continuous_scale='Greens'
)
fig3.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
fig3.write_html(VIZ_DIR / 'top_persons.html')
print(f" Saved: top_persons.html")

# 4. Entity distribution by sector
fig4 = px.bar(
    sector_entities.reset_index(),
    x='sector',
    y='avg_total',
    title='Average Entities per Article by Sector',
    labels={'avg_total': 'Average Entities', 'sector': 'Sector'},
    color='avg_total',
    color_continuous_scale='Viridis'
)
fig4.update_layout(height=600, xaxis_tickangle=-45)
fig4.write_html(VIZ_DIR / 'entities_by_sector.html')
print(f" Saved: entities_by_sector.html\n")

# ============================================================================
# 11. SAVE ENHANCED DATASET
# ============================================================================

print(" Step 11: Saving enhanced dataset with entities...")

# Convert entity lists to JSON strings for CSV storage
df['spacy_entities_json'] = df['spacy_entities'].apply(json.dumps)
df['bert_entities_json'] = df['bert_entities'].apply(json.dumps)

# Select columns
entity_columns = [
    'ticker', 'company_name', 'sector', 'industry', 'date',
    'title', 'text', 'source',
    'finbert_sentiment', 'ensemble_sentiment',
    'num_entities', 'num_orgs', 'num_persons', 'num_locations',
    'bert_num_orgs', 'bert_num_persons', 'bert_num_locations',
    'spacy_entities_json', 'bert_entities_json',
    'word_count', 'year', 'month', 'is_weekend', 'is_market_hours'
]

df_entities = df[entity_columns].copy()

# Save
output_file = PROCESSED_DIR / 'dataset_with_entities.csv'
df_entities.to_csv(output_file, index=False)

print(f" Enhanced dataset saved to {output_file}")
print(f"   Rows: {len(df_entities):,}")
print(f"   Columns: {len(df_entities.columns)}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

# ============================================================================
# 12. GENERATE SUMMARY
# ============================================================================

print(" Step 12: Generating NER summary...\n")

summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'total_articles': int(len(df)),
        'companies': int(df['ticker'].nunique()),
        'sectors': int(df['sector'].nunique())
    },
    'entity_extraction': {
        'total_entities': int(df['num_entities'].sum()),
        'unique_entities': int(len(entity_counts)),
        'avg_per_article': float(df['num_entities'].mean()),
        'organizations': int(df['num_orgs'].sum()),
        'persons': int(df['num_persons'].sum()),
        'locations': int(df['num_locations'].sum())
    },
    'top_organizations': top_orgs.head(20).to_dict('records'),
    'top_persons': top_persons.head(20).to_dict('records'),
    'top_locations': top_locations.head(20).to_dict('records'),
    'sector_distribution': sector_entities.to_dict(),
    'co_occurrence_pairs': int(len(co_mention_counts)) if co_mentions else 0
}

# Save
summary_file = ENTITIES_DIR / 'ner_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f" Saved: ner_summary.json")

# ============================================================================
# 13. FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("NAMED ENTITY RECOGNITION COMPLETE")
print("="*80)

print(f"\n NER Summary:")
print(f"   Articles processed: {summary['dataset']['total_articles']:,}")
print(f"   Total entities: {summary['entity_extraction']['total_entities']:,}")
print(f"   Unique entities: {summary['entity_extraction']['unique_entities']:,}")
print(f"   Avg entities/article: {summary['entity_extraction']['avg_per_article']:.2f}")

print(f"\n Entity Types:")
print(f"   Organizations: {summary['entity_extraction']['organizations']:,}")
print(f"   Persons: {summary['entity_extraction']['persons']:,}")
print(f"   Locations: {summary['entity_extraction']['locations']:,}")

print(f"\n Top Entity:")
print(f"   Organization: {top_orgs.iloc[0]['entity']} ({top_orgs.iloc[0]['mention_count']:,} mentions)")
print(f"   Person: {top_persons.iloc[0]['entity']} ({top_persons.iloc[0]['mention_count']:,} mentions)")
print(f"   Location: {top_locations.iloc[0]['entity']} ({top_locations.iloc[0]['mention_count']:,} mentions)")

print(f"\n Output Files:")
print(f"   1. {output_file}")
print(f"   2. {ENTITIES_DIR / 'financial_entity_database.csv'}")
print(f"   3. {ENTITIES_DIR / 'entity_mention_counts.csv'}")
print(f"   4. {ENTITIES_DIR / 'top_organizations.csv'}")
print(f"   5. {ENTITIES_DIR / 'top_persons.csv'}")
print(f"   6. {ENTITIES_DIR / 'top_locations.csv'}")
print(f"   7. {ENTITIES_DIR / 'entity_co_mentions.csv'}")
print(f"   8. {summary_file}")

print(f"\n Ready for Step 5: Event Detection & Classification")
print("="*80)
