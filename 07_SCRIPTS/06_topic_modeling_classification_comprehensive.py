"""
Topic Modeling and Classification - Comprehensive
=================================================

Latent Dirichlet Allocation (LDA) for topic discovery
Uses Gensim for robust topic modeling

Target: Topic coherence >0.45, 10 interpretable topics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP and Topic Modeling
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from tqdm.auto import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('seaborn-v0_8-darkgrid')

# Download NLTK data
for package in ['stopwords', 'wordnet', 'punkt', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package, quiet=True)

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / '01_DATA'
PROCESSED_DIR = DATA_DIR / 'processed'
TOPICS_DIR = DATA_DIR / 'topics'
MODELS_DIR = BASE_DIR / '04_MODELS' / 'topic_models'
VIZ_DIR = DATA_DIR / 'visualizations'

# Create directories
TOPICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TOPIC MODELING & CLASSIFICATION - COMPREHENSIVE")
print("="*80)
print(f" Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print(" Step 1: Loading dataset with events...")

df = pd.read_csv(PROCESSED_DIR / 'dataset_with_events.csv')
df['date'] = pd.to_datetime(df['date'])

print(f" Loaded {len(df):,} articles")
print(f"   Companies: {df['ticker'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}\n")

# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================

print(" Step 2: Preprocessing text for topic modeling...")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add financial stop words
        self.stop_words.update([
            'said', 'would', 'could', 'also', 'may', 'might', 'must',
            'stock', 'share', 'company', 'market', 'price', 'trading',
            'new', 'year', 'day', 'time', 'week', 'month'
        ])
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        """Clean and preprocess text"""
        # Tokenize
        tokens = word_tokenize(str(text).lower())
        
        # Remove stopwords, short words, numbers
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and 
               len(token) > 3 and
               token not in self.stop_words
        ]
        
        return tokens

preprocessor = TextPreprocessor()

print("   Preprocessing documents...")
processed_texts = [preprocessor.preprocess(text) for text in tqdm(df['text'], desc="Preprocessing")]

# Filter out empty documents
valid_indices = [i for i, tokens in enumerate(processed_texts) if len(tokens) >= 10]
processed_texts = [processed_texts[i] for i in valid_indices]
df_valid = df.iloc[valid_indices].reset_index(drop=True)

print(f"\n Text preprocessing complete")
print(f"   Valid documents: {len(processed_texts):,}")
print(f"   Average tokens per document: {np.mean([len(t) for t in processed_texts]):.1f}\n")

# ============================================================================
# 3. CREATE DICTIONARY AND CORPUS
# ============================================================================

print(" Step 3: Creating dictionary and corpus...")

# Create dictionary
dictionary = corpora.Dictionary(processed_texts)

# Filter extremes
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=5000)

# Create corpus
corpus = [dictionary.doc2bow(text) for text in processed_texts]

print(f" Dictionary and corpus created")
print(f"   Vocabulary size: {len(dictionary):,} terms")
print(f"   Corpus size: {len(corpus):,} documents")
print(f"   Average terms per document: {np.mean([len(doc) for doc in corpus]):.1f}\n")

# Save dictionary and corpus
dictionary.save(str(MODELS_DIR / 'dictionary.gensim'))
corpora.MmCorpus.serialize(str(MODELS_DIR / 'corpus.mm'), corpus)
print(f" Saved: dictionary.gensim")
print(f" Saved: corpus.mm\n")

# ============================================================================
# 4. TRAIN LDA MODEL
# ============================================================================

print(" Step 4: Training LDA model...")

# LDA parameters
num_topics = 10
passes = 10
iterations = 100
alpha = 'auto'
eta = 'auto'

print(f"   Parameters:")
print(f"   - Number of topics: {num_topics}")
print(f"   - Passes: {passes}")
print(f"   - Iterations: {iterations}")
print(f"   - Alpha: {alpha}")
print(f"   - Eta: {eta}\n")

# Train LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=passes,
    iterations=iterations,
    alpha=alpha,
    eta=eta,
    per_word_topics=True
)

print(f" LDA model trained\n")

# Save model
lda_model.save(str(MODELS_DIR / 'lda_model.gensim'))
print(f" Saved: lda_model.gensim\n")

# ============================================================================
# 5. CALCULATE COHERENCE SCORE
# ============================================================================

print(" Step 5: Calculating coherence score...")

coherence_model = CoherenceModel(
    model=lda_model,
    texts=processed_texts,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model.get_coherence()

print(f" Coherence score: {coherence_score:.4f}")
print(f"   Target: >0.45")
print(f"   Status: {' MEETS TARGET' if coherence_score >= 0.45 else ' BELOW TARGET'}\n")

# ============================================================================
# 6. EXTRACT TOPIC KEYWORDS
# ============================================================================

print(" Step 6: Extracting topic keywords...\n")

# Get top keywords for each topic
topic_keywords = []
for topic_id in range(num_topics):
    words = lda_model.show_topic(topic_id, topn=15)
    keywords = [word for word, prob in words]
    topic_keywords.append({
        'topic_id': topic_id,
        'keywords': ', '.join(keywords[:10]),
        'top_words': keywords[:5]
    })

topics_df = pd.DataFrame(topic_keywords)

print(" Topic Keywords:")
for idx, row in topics_df.iterrows():
    print(f"   Topic {row['topic_id']}: {row['keywords']}")
print()

# Save
topics_df.to_csv(TOPICS_DIR / 'topic_keywords.csv', index=False)
print(f" Saved: topic_keywords.csv\n")

# ============================================================================
# 7. ASSIGN TOPICS TO DOCUMENTS
# ============================================================================

print(" Step 7: Assigning topics to documents...")

def get_document_topics(lda_model, corpus):
    """Get dominant topic for each document"""
    doc_topics = []
    
    for doc_bow in tqdm(corpus, desc="Assigning topics"):
        topic_dist = lda_model.get_document_topics(doc_bow)
        
        if topic_dist:
            # Get dominant topic
            dominant_topic = max(topic_dist, key=lambda x: x[1])
            
            # Get all topic probabilities
            topic_probs = {f'topic_{t}': p for t, p in topic_dist}
            
            doc_topics.append({
                'dominant_topic': dominant_topic[0],
                'topic_probability': dominant_topic[1],
                **topic_probs
            })
        else:
            doc_topics.append({
                'dominant_topic': -1,
                'topic_probability': 0.0
            })
    
    return doc_topics

doc_topics = get_document_topics(lda_model, corpus)

# Add to dataframe
df_valid['dominant_topic'] = [dt['dominant_topic'] for dt in doc_topics]
df_valid['topic_probability'] = [dt['topic_probability'] for dt in doc_topics]

# Add topic keywords
df_valid['topic_keywords'] = df_valid['dominant_topic'].apply(
    lambda x: topics_df.loc[x, 'keywords'] if x >= 0 else ''
)

# Add probability columns
for i in range(num_topics):
    col_name = f'topic_{i}_prob'
    df_valid[col_name] = [dt.get(col_name, 0.0) for dt in doc_topics]

print(f"\n Topics assigned to documents")
print(f"   Topic distribution:")
print(df_valid['dominant_topic'].value_counts().sort_index())
print()

# ============================================================================
# 8. TOPIC DISTRIBUTION ANALYSIS
# ============================================================================

print(" Step 8: Analyzing topic distribution...\n")

topic_dist = df_valid['dominant_topic'].value_counts().sort_index()

print(" Topic Distribution:")
for topic_id, count in topic_dist.items():
    pct = (count / len(df_valid)) * 100
    keywords = topics_df.loc[topic_id, 'keywords']
    print(f"   Topic {topic_id}: {count:5,} ({pct:5.1f}%) - {keywords}")
print()

# ============================================================================
# 9. TOPIC-SECTOR ANALYSIS
# ============================================================================

print(" Step 9: Analyzing topics by sector...")

topic_sector = df_valid.groupby(['sector', 'dominant_topic']).size().reset_index(name='count')
topic_sector_pivot = topic_sector.pivot(index='sector', columns='dominant_topic', values='count').fillna(0)

print(f" Topic-sector analysis complete\n")

# Save
topic_sector_pivot.to_csv(TOPICS_DIR / 'topic_sector_distribution.csv')
print(f" Saved: topic_sector_distribution.csv\n")

# ============================================================================
# 10. TEMPORAL TOPIC TRENDS
# ============================================================================

print(" Step 10: Analyzing temporal topic trends...")

df_valid['year_month'] = df_valid['date'].dt.to_period('M').astype(str)

temporal_topics = df_valid.groupby(['year_month', 'dominant_topic']).size().reset_index(name='count')
temporal_topics_pivot = temporal_topics.pivot(index='year_month', columns='dominant_topic', values='count').fillna(0)

print(f" Temporal topic trends calculated")
print(f"   Months covered: {len(temporal_topics_pivot)}\n")

# Save
temporal_topics_pivot.to_csv(TOPICS_DIR / 'temporal_topic_trends.csv')
print(f" Saved: temporal_topic_trends.csv\n")

# ============================================================================
# 11. TOPIC-SENTIMENT CORRELATION
# ============================================================================

print(" Step 11: Analyzing topic-sentiment correlation...")

topic_sentiment = df_valid.groupby('dominant_topic').agg({
    'finbert_sentiment': ['mean', 'std'],
    'ensemble_sentiment': ['mean', 'std'],
    'ticker': 'count'
}).round(4)

topic_sentiment.columns = ['finbert_mean', 'finbert_std', 'ensemble_mean', 'ensemble_std', 'doc_count']
topic_sentiment = topic_sentiment.reset_index()

print(f" Topic-Sentiment Correlation:\n")
print(topic_sentiment)
print()

# Save
topic_sentiment.to_csv(TOPICS_DIR / 'topic_sentiment_correlation.csv', index=False)
print(f" Saved: topic_sentiment_correlation.csv\n")

# ============================================================================
# 12. VISUALIZATIONS
# ============================================================================

print(" Step 12: Creating visualizations...\n")

# 1. Topic distribution
fig1 = px.bar(
    x=topic_dist.index,
    y=topic_dist.values,
    title='Topic Distribution',
    labels={'x': 'Topic ID', 'y': 'Number of Documents'},
    color=topic_dist.values,
    color_continuous_scale='Viridis'
)
fig1.update_layout(height=500, showlegend=False)
fig1.write_html(VIZ_DIR / 'topic_distribution.html')
print(f" Saved: topic_distribution.html")

# 2. Topics over time
fig2 = go.Figure()
for topic_id in range(num_topics):
    fig2.add_trace(go.Scatter(
        x=temporal_topics_pivot.index,
        y=temporal_topics_pivot[topic_id],
        mode='lines',
        name=f'Topic {topic_id}',
        stackgroup='one'
    ))
fig2.update_layout(
    title='Topic Evolution Over Time (Stacked)',
    xaxis_title='Month',
    yaxis_title='Number of Documents',
    height=600,
    xaxis_tickangle=-45,
    hovermode='x unified'
)
fig2.write_html(VIZ_DIR / 'topic_evolution.html')
print(f" Saved: topic_evolution.html")

# 3. Topic-sentiment correlation
fig3 = px.bar(
    topic_sentiment,
    x='dominant_topic',
    y='ensemble_mean',
    title='Average Sentiment by Topic',
    labels={'dominant_topic': 'Topic ID', 'ensemble_mean': 'Average Sentiment'},
    color='ensemble_mean',
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0,
    error_y='ensemble_std'
)
fig3.add_hline(y=0, line_dash="dash", line_color="gray")
fig3.update_layout(height=500)
fig3.write_html(VIZ_DIR / 'topic_sentiment.html')
print(f" Saved: topic_sentiment.html\n")

# ============================================================================
# 13. SAVE ENHANCED DATASET
# ============================================================================

print(" Step 13: Saving enhanced dataset with topics...")

# Select columns
topic_columns = [
    'ticker', 'company_name', 'sector', 'industry', 'date',
    'title', 'text', 'source',
    'finbert_sentiment', 'ensemble_sentiment',
    'num_entities', 'num_events', 'event_impact_score',
    'dominant_topic', 'topic_probability', 'topic_keywords',
    'word_count', 'year', 'month', 'is_weekend', 'is_market_hours'
]

# Add topic probability columns
for i in range(num_topics):
    topic_columns.append(f'topic_{i}_prob')

df_topics = df_valid[topic_columns].copy()

# Save
output_file = PROCESSED_DIR / 'dataset_with_topics.csv'
df_topics.to_csv(output_file, index=False)

print(f" Enhanced dataset saved to {output_file}")
print(f"   Rows: {len(df_topics):,}")
print(f"   Columns: {len(df_topics.columns)}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")

# ============================================================================
# 14. GENERATE SUMMARY
# ============================================================================

summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_parameters': {
        'num_topics': num_topics,
        'passes': passes,
        'iterations': iterations,
        'vocabulary_size': len(dictionary),
        'corpus_size': len(corpus)
    },
    'performance': {
        'coherence_score': float(coherence_score),
        'meets_target': bool(coherence_score >= 0.45)
    },
    'topic_distribution': topic_dist.to_dict(),
    'topics': topic_keywords,
    'topic_sentiment': topic_sentiment.to_dict('records')
}

summary_file = TOPICS_DIR / 'topic_modeling_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f" Saved: topic_modeling_summary.json")

# ============================================================================
# 15. FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("TOPIC MODELING & CLASSIFICATION COMPLETE")
print("="*80)

print(f"\n Topic Modeling Summary:")
print(f"   Documents analyzed: {len(df_topics):,}")
print(f"   Number of topics: {num_topics}")
print(f"   Vocabulary size: {len(dictionary):,}")
print(f"   Coherence score: {coherence_score:.4f}")
print(f"   Status: {' MEETS TARGET (>0.45)' if coherence_score >= 0.45 else ' BELOW TARGET'}")

print(f"\n Topic Distribution:")
for topic_id in range(min(5, num_topics)):
    count = topic_dist.get(topic_id, 0)
    pct = (count / len(df_topics)) * 100
    keywords = topics_df.loc[topic_id, 'top_words']
    print(f"   Topic {topic_id}: {count:,} docs ({pct:.1f}%) - {', '.join(keywords)}")

print(f"\n Output Files:")
print(f"   1. {output_file}")
print(f"   2. {MODELS_DIR / 'lda_model.gensim'}")
print(f"   3. {MODELS_DIR / 'dictionary.gensim'}")
print(f"   4. {TOPICS_DIR / 'topic_keywords.csv'}")
print(f"   5. {TOPICS_DIR / 'topic_sentiment_correlation.csv'}")
print(f"   6. {summary_file}")

print(f"\n Ready for Step 7: Advanced Feature Engineering")
print("="*80)
