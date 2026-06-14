"""
Ensemble Modeling Strategies - Comprehensive
============================================

Trains and evaluates 8 base models + 4 ensemble methods:

Base Models:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. LightGBM
6. SVM
7. Naive Bayes
8. KNN

Ensemble Methods:
1. Hard Voting
2. Soft Voting
3. Bagging
4. Stacking

Target: 88.9% accuracy matching institutional benchmarks
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Base models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
FEATURES_DIR = DATA_DIR / 'features'
MODELS_DIR = BASE_DIR / '04_MODELS'
RESULTS_DIR = BASE_DIR / '03_RESULTS'
VIZ_DIR = RESULTS_DIR / 'visualizations'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / 'base_models').mkdir(parents=True, exist_ok=True)
(MODELS_DIR / 'ensemble').mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ENSEMBLE MODELING STRATEGIES - COMPREHENSIVE")
print("="*80)
print(f" Base directory: {BASE_DIR}\n")

# ============================================================================
# 1. LOAD FEATURE MATRIX
# ============================================================================

print(" Step 1: Loading final feature matrix...")

df = pd.read_csv(FEATURES_DIR / 'final_feature_matrix.csv')
df['date'] = pd.to_datetime(df['date'])

print(f" Loaded {len(df):,} articles")
print(f"   Features: {df.shape[1]}")
print(f"   Companies: {df['ticker'].nunique()}\n")

# ============================================================================
# 2. PREPARE DATA FOR MODELING
# ============================================================================

print(" Step 2: Preparing data for modeling...")

# Create target variable (3-class sentiment)
df['sentiment_class'] = pd.cut(
    df['finbert_sentiment'],
    bins=[-1.1, -0.1, 0.1, 1.1],
    labels=['negative', 'neutral', 'positive']
)

# Select features
feature_cols = [
    # Sentiment
    'sentiment_momentum', 'sentiment_ma_7d', 'sentiment_extremity', 'sentiment_consensus',
    # Entity
    'entity_density', 'num_entities', 'entity_diversity',
    # Event
    'event_impact_score', 'has_high_impact_event', 'event_density',
    # Topic
    'topic_dominance', 'topic_entropy',
    # Temporal
    'is_market_hours', 'days_since_last', 'recency_score',
    # Velocity
    'article_count_7d', 'news_burst', 'news_acceleration',
    # Complexity
    'word_count', 'readability_score',
    # Composite
    'news_importance', 'attention_score', 'signal_strength', 'market_relevance',
    # Additional
    'sentiment_volatility', 'entity_momentum', 'event_momentum'
]

# Remove any missing features
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].copy()
y = df['sentiment_class'].copy()

# Remove rows with missing target
valid_mask = y.notna()
X = X[valid_mask]
y = y[valid_mask]

print(f" Data prepared")
print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X):,}")
print(f"   Classes: {y.nunique()}")
print(f"   Class distribution:")
print(y.value_counts())
print()

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

print(" Step 3: Creating train-test split...")

# Time-aware split (no shuffle)
df_with_target = df[valid_mask].reset_index(drop=True)
df_with_target = df_with_target.sort_values('date')

X_sorted = df_with_target[feature_cols]
y_sorted = df_with_target['sentiment_class']

split_idx = int(len(X_sorted) * 0.8)
X_train = X_sorted.iloc[:split_idx]
X_test = X_sorted.iloc[split_idx:]
y_train = y_sorted.iloc[:split_idx]
y_test = y_sorted.iloc[split_idx:]

print(f" Train-test split created")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Train period: {df_with_target.iloc[:split_idx]['date'].min()} to {df_with_target.iloc[:split_idx]['date'].max()}")
print(f"   Test period: {df_with_target.iloc[split_idx:]['date'].min()} to {df_with_target.iloc[split_idx:]['date'].max()}\n")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================

print(" Step 4: Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Features scaled using StandardScaler\n")

# Save scaler
joblib.dump(scaler, MODELS_DIR / 'feature_scaler.pkl')
print(f" Saved: feature_scaler.pkl\n")

# ============================================================================
# 5. TRAIN BASE MODELS
# ============================================================================

print(" Step 5: Training 8 base models...\n")

base_models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42, eval_metric='mlogloss', use_label_encoder=False),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42, verbose=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance')
}

results = {}

for name, model in tqdm(base_models.items(), desc="Training base models"):
    print(f"\n   Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
    
    print(f"      Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save model
    model_file = MODELS_DIR / 'base_models' / f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_file)

print(f"\n All base models trained and saved\n")

# ============================================================================
# 6. BASE MODEL COMPARISON
# ============================================================================

print(" Step 6: Comparing base model performance...\n")

performance_df = pd.DataFrame({
    'Model': results.keys(),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()]
})

performance_df = performance_df.sort_values('Accuracy', ascending=False)

print(" Base Model Performance:")
print("="*90)
print(performance_df.to_string(index=False))
print("="*90)
print()

# Save
performance_df.to_csv(RESULTS_DIR / 'base_model_performance.csv', index=False)
print(f" Saved: base_model_performance.csv\n")

# ============================================================================
# 7. ENSEMBLE METHOD 1: HARD VOTING
# ============================================================================

print(" Step 7: Creating Hard Voting ensemble...")

# Use top 3 models
top_3_models = performance_df.head(3)['Model'].tolist()
voting_estimators = [(name, results[name]['model']) for name in top_3_models]

hard_voting = VotingClassifier(estimators=voting_estimators, voting='hard')
hard_voting.fit(X_train_scaled, y_train)
y_pred_hard = hard_voting.predict(X_test_scaled)

acc_hard = accuracy_score(y_test, y_pred_hard)
f1_hard = f1_score(y_test, y_pred_hard, average='weighted')

print(f" Hard Voting Ensemble")
print(f"   Models: {', '.join(top_3_models)}")
print(f"   Accuracy: {acc_hard:.4f}, F1: {f1_hard:.4f}\n")

# Save
joblib.dump(hard_voting, MODELS_DIR / 'ensemble' / 'hard_voting.pkl')

# ============================================================================
# 8. ENSEMBLE METHOD 2: SOFT VOTING
# ============================================================================

print(" Step 8: Creating Soft Voting ensemble...")

# Use all models with predict_proba
soft_estimators = [
    (name, results[name]['model']) 
    for name in results.keys() 
    if hasattr(results[name]['model'], 'predict_proba')
]

soft_voting = VotingClassifier(estimators=soft_estimators, voting='soft')
soft_voting.fit(X_train_scaled, y_train)
y_pred_soft = soft_voting.predict(X_test_scaled)

acc_soft = accuracy_score(y_test, y_pred_soft)
f1_soft = f1_score(y_test, y_pred_soft, average='weighted')

print(f" Soft Voting Ensemble")
print(f"   Models: {len(soft_estimators)}")
print(f"   Accuracy: {acc_soft:.4f}, F1: {f1_soft:.4f}\n")

# Save
joblib.dump(soft_voting, MODELS_DIR / 'ensemble' / 'soft_voting.pkl')

# ============================================================================
# 9. ENSEMBLE METHOD 3: BAGGING
# ============================================================================

print(" Step 9: Creating Bagging ensemble...")

# Best base model
best_model_name = performance_df.iloc[0]['Model']
best_base_model = results[best_model_name]['model']

bagging = BaggingClassifier(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

bagging.fit(X_train_scaled, y_train)
y_pred_bagging = bagging.predict(X_test_scaled)

acc_bagging = accuracy_score(y_test, y_pred_bagging)
f1_bagging = f1_score(y_test, y_pred_bagging, average='weighted')

print(f" Bagging Ensemble")
print(f"   Base model: Random Forest")
print(f"   Estimators: 10")
print(f"   Accuracy: {acc_bagging:.4f}, F1: {f1_bagging:.4f}\n")

# Save
joblib.dump(bagging, MODELS_DIR / 'ensemble' / 'bagging.pkl')

# ============================================================================
# 10. ENSEMBLE METHOD 4: STACKING
# ============================================================================

print(" Step 10: Creating Stacking ensemble...")

# Use top 5 models as base learners
top_5_models = performance_df.head(5)['Model'].tolist()
stacking_estimators = [(name, results[name]['model']) for name in top_5_models]

meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

stacking = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=meta_learner,
    cv=5
)

stacking.fit(X_train_scaled, y_train)
y_pred_stacking = stacking.predict(X_test_scaled)

acc_stacking = accuracy_score(y_test, y_pred_stacking)
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')

print(f" Stacking Ensemble")
print(f"   Base models: {len(stacking_estimators)}")
print(f"   Meta-learner: Logistic Regression")
print(f"   Accuracy: {acc_stacking:.4f}, F1: {f1_stacking:.4f}\n")

# Save
joblib.dump(stacking, MODELS_DIR / 'ensemble' / 'stacking.pkl')

# ============================================================================
# 11. ENSEMBLE COMPARISON
# ============================================================================

print(" Step 11: Comparing all models...\n")

ensemble_results = pd.DataFrame({
    'Model': list(results.keys()) + ['Hard Voting', 'Soft Voting', 'Bagging', 'Stacking'],
    'Accuracy': [r['accuracy'] for r in results.values()] + [acc_hard, acc_soft, acc_bagging, acc_stacking],
    'F1-Score': [r['f1'] for r in results.values()] + [f1_hard, f1_soft, f1_bagging, f1_stacking]
})

ensemble_results = ensemble_results.sort_values('Accuracy', ascending=False)

print(" Complete Model Performance Comparison:")
print("="*70)
print(ensemble_results.to_string(index=False))
print("="*70)
print()

# Save
ensemble_results.to_csv(RESULTS_DIR / 'all_model_performance.csv', index=False)
print(f" Saved: all_model_performance.csv\n")

# ============================================================================
# 12. BEST MODEL ANALYSIS
# ============================================================================

best_model = ensemble_results.iloc[0]['Model']
best_accuracy = ensemble_results.iloc[0]['Accuracy']

print(f" Best Model: {best_model}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   Target: 0.889")
print(f"   Status: {' MEETS TARGET' if best_accuracy >= 0.889 else ' BELOW TARGET'}\n")

# ============================================================================
# 13. VISUALIZATIONS
# ============================================================================

print(" Step 13: Creating visualizations...\n")

# Model comparison
fig1 = px.bar(
    ensemble_results,
    x='Model',
    y='Accuracy',
    title='Model Performance Comparison',
    labels={'Accuracy': 'Accuracy Score'},
    color='Accuracy',
    color_continuous_scale='Viridis'
)
fig1.add_hline(y=0.889, line_dash="dash", line_color="red", annotation_text="Target: 0.889")
fig1.update_layout(height=600, xaxis_tickangle=-45, showlegend=False)
fig1.write_html(VIZ_DIR / 'model_comparison.html')
print(f" Saved: model_comparison.html")

# Confusion matrix for best model
best_pred = y_pred_stacking if best_model == 'Stacking' else results[best_model]['predictions']
cm = confusion_matrix(y_test, best_pred)

fig2 = px.imshow(
    cm,
    labels=dict(x="Predicted", y="Actual"),
    x=['negative', 'neutral', 'positive'],
    y=['negative', 'neutral', 'positive'],
    text_auto=True,
    title=f'Confusion Matrix - {best_model}',
    color_continuous_scale='Blues'
)
fig2.write_html(VIZ_DIR / 'confusion_matrix_best.html')
print(f" Saved: confusion_matrix_best.html\n")

# ============================================================================
# 14. FINAL SUMMARY
# ============================================================================

print("="*80)
print("ENSEMBLE MODELING COMPLETE")
print("="*80)

print(f"\n Modeling Summary:")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Features: {len(feature_cols)}")
print(f"   Base models: 8")
print(f"   Ensemble methods: 4")
print(f"   Total models: 12")

print(f"\n Best Performing Model:")
print(f"   Name: {best_model}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   F1-Score: {ensemble_results.iloc[0]['F1-Score']:.4f}")

print(f"\n Top 3 Models:")
for idx, row in ensemble_results.head(3).iterrows():
    print(f"   {row['Model']}: {row['Accuracy']:.4f}")

print(f"\n Output Files:")
print(f"   Models: {len(list((MODELS_DIR / 'base_models').glob('*.pkl')))} base + {len(list((MODELS_DIR / 'ensemble').glob('*.pkl')))} ensemble")
print(f"   Results: {RESULTS_DIR / 'all_model_performance.csv'}")
print(f"   Scaler: {MODELS_DIR / 'feature_scaler.pkl'}")

print(f"\n Ready for Step 9: Trading Signal Generation")
print("="*80)
