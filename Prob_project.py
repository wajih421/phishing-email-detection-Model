

# ─────────────────────────────────────────
#   Importing Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r"D:\UNI\SEM-6\PROB AND STATS\PROJECT")

# Plot style
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor']   = '#ffffff'
plt.rcParams['font.family']      = 'DejaVu Sans'

print("=" * 60)
print("  Phishing Email Detection - Statistical Analysis")
print("=" * 60)

# ─────────────────────────────────────────
#   Loading Dataset (2000 samples)
# ─────────────────────────────────────────
print("\n[STEP 2] Loading Dataset...")


df_full = pd.read_csv('spam_email_dataset.csv')
# Use 2000 samples: 1200 legitimate + 800 phishing
legit   = df_full[df_full['label'] == 0].sample(n=1200, random_state=42)
phish   = df_full[df_full['label'] == 1].sample(n=800,  random_state=42)
df      = pd.concat([legit, phish]).reset_index(drop=True)
df      = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"  Total samples used : {len(df)}")
print(f"  Legitimate emails  : {len(legit)}  (60%)")
print(f"  Phishing emails    : {len(phish)}  (40%)")
print(f"  Null values        : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
#  Feature Selection
# ─────────────────────────────────────────
print("\n[STEP 3] Selecting Features...")

FEATURES = [
    'num_words',
    'num_characters',
    'num_exclamation_marks',
    'num_links',
    'has_suspicious_link',
    'num_attachments',
    'has_attachment',
    'sender_reputation_score',
    'num_recipients',
    'contains_money_terms',
    'contains_urgency_terms'
]
LABEL = 'label'

X = df[FEATURES]
y = df[LABEL]

print(f"  Features selected  : {len(FEATURES)}")
print(f"  Feature names      : {FEATURES}")

# ─────────────────────────────────────────
#   Statistical Analysis
# ─────────────────────────────────────────
print("\n[STEP 4] Statistical Analysis...")

legit_df = df[df[LABEL] == 0]
phish_df = df[df[LABEL] == 1]

stats_rows = []
for feat in FEATURES:
    stats_rows.append({
        'Feature'           : feat,
        'Legit Mean'        : round(legit_df[feat].mean(), 3),
        'Legit Std'         : round(legit_df[feat].std(),  3),
        'Phishing Mean'     : round(phish_df[feat].mean(), 3),
        'Phishing Std'      : round(phish_df[feat].std(),  3),
    })

stats_df = pd.DataFrame(stats_rows)
print("\n  Feature Statistics (Mean & Std Dev):")
print(stats_df.to_string(index=False))

# ─────────────────────────────────────────
#   Probability Analysis
# ─────────────────────────────────────────
print("\n[STEP 5] Probability Analysis...")

P_phish  = len(phish_df) / len(df)
P_legit  = len(legit_df) / len(df)

P_money_given_phish  = phish_df['contains_money_terms'].mean()
P_money_given_legit  = legit_df['contains_money_terms'].mean()
P_urgency_given_phish = phish_df['contains_urgency_terms'].mean()
P_urgency_given_legit = legit_df['contains_urgency_terms'].mean()
P_suslink_given_phish = phish_df['has_suspicious_link'].mean()
P_suslink_given_legit = legit_df['has_suspicious_link'].mean()

print(f"\n  P(Phishing)                        = {P_phish:.4f}")
print(f"  P(Legitimate)                      = {P_legit:.4f}")
print(f"\n  P(Money Terms | Phishing)          = {P_money_given_phish:.4f}")
print(f"  P(Money Terms | Legitimate)        = {P_money_given_legit:.4f}")
print(f"\n  P(Urgency Terms | Phishing)        = {P_urgency_given_phish:.4f}")
print(f"  P(Urgency Terms | Legitimate)      = {P_urgency_given_legit:.4f}")
print(f"\n  P(Suspicious Link | Phishing)      = {P_suslink_given_phish:.4f}")
print(f"  P(Suspicious Link | Legitimate)    = {P_suslink_given_legit:.4f}")

# ─────────────────────────────────────────
# Train / Test Split & Model
# ─────────────────────────────────────────
print("\n[STEP 6] Training Naive Bayes Model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"  Training samples   : {len(X_train)}")
print(f"  Testing samples    : {len(X_test)}")

# ─────────────────────────────────────────
#  Performance Metrics
# ─────────────────────────────────────────
print("\n[STEP 7] Model Performance...")

acc       = accuracy_score(y_test, y_pred)
prec      = precision_score(y_test, y_pred)
rec       = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)   # False Positive Rate
fnr = fn / (fn + tp)   # False Negative Rate

print(f"\n  Accuracy           : {acc*100:.2f}%")
print(f"  Precision          : {prec*100:.2f}%")
print(f"  Recall             : {rec*100:.2f}%")
print(f"  F1-Score           : {f1*100:.2f}%")
print(f"\n  True Positives     : {tp}  (Phishing correctly detected)")
print(f"  True Negatives     : {tn}  (Legit correctly classified)")
print(f"  False Positives    : {fp}  (Legit wrongly flagged as Phishing)")
print(f"  False Negatives    : {fn}  (Phishing missed)")
print(f"\n  False Positive Rate: {fpr*100:.2f}%")
print(f"  False Negative Rate: {fnr*100:.2f}%")

print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Phishing']))


# ╔══════════════════════════════════════════════════════════╗
#                    GRAPHS SECTION
# ╚══════════════════════════════════════════════════════════╝

COLORS = {'legit': '#2196F3', 'phish': '#F44336',
          'accent': '#FF9800', 'green': '#4CAF50'}

# ─────────────────────────────────────────
#  GRAPH 1: Class Distribution (Pie + Bar)
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Graph 1 — Dataset Class Distribution', fontsize=15, fontweight='bold', y=1.02)

# Pie chart
labels_pie = ['Legitimate (0)', 'Phishing (1)']
sizes      = [len(legit_df), len(phish_df)]
colors_pie = [COLORS['legit'], COLORS['phish']]
explode    = (0.05, 0.05)
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=140, explode=explode,
    textprops={'fontsize': 12})
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title('Proportion of Email Classes', fontsize=12, fontweight='bold')

# Bar chart
bars = axes[1].bar(['Legitimate', 'Phishing'], sizes,
                   color=[COLORS['legit'], COLORS['phish']],
                   edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 15,
                 str(val), ha='center', fontsize=12, fontweight='bold')
axes[1].set_title('Email Count by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Emails', fontsize=11)
axes[1].set_ylim(0, max(sizes) * 1.15)
axes[1].grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('graph1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[GRAPH 1] Saved: class distribution")


# ─────────────────────────────────────────
#   Feature Distribution Histograms
#           (4 key numerical features)
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Graph 2 — Feature Distributions: Phishing vs Legitimate',
             fontsize=15, fontweight='bold')

feat_titles = {
    'num_words'             : 'Number of Words',
    'num_exclamation_marks' : 'Exclamation Marks Count',
    'num_links'             : 'Number of Links',
    'sender_reputation_score': 'Sender Reputation Score'
}

for ax, (feat, title) in zip(axes.flat, feat_titles.items()):
    ax.hist(legit_df[feat], bins=25, alpha=0.65,
            color=COLORS['legit'], label='Legitimate', edgecolor='white')
    ax.hist(phish_df[feat], bins=25, alpha=0.65,
            color=COLORS['phish'], label='Phishing', edgecolor='white')
    ax.axvline(legit_df[feat].mean(), color=COLORS['legit'],
               linestyle='--', linewidth=2,
               label=f'Legit Mean={legit_df[feat].mean():.2f}')
    ax.axvline(phish_df[feat].mean(), color=COLORS['phish'],
               linestyle='--', linewidth=2,
               label=f'Phish Mean={phish_df[feat].mean():.2f}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graph2_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 2] Saved: feature distributions")


# ─────────────────────────────────────────
#  Box Plots Comparison
# ─────────────────────────────────────────
box_features = ['num_words', 'num_links',
                'num_exclamation_marks', 'sender_reputation_score']

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
fig.suptitle('Graph 3 — Box Plots: Phishing vs Legitimate',
             fontsize=15, fontweight='bold')

for ax, feat in zip(axes, box_features):
    data_to_plot = [legit_df[feat].values, phish_df[feat].values]
    bp = ax.boxplot(data_to_plot, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(COLORS['legit'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLORS['phish'])
    bp['boxes'][1].set_alpha(0.7)
    ax.set_xticklabels(['Legitimate', 'Phishing'], fontsize=10)
    ax.set_title(feat.replace('_', '\n'), fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graph3_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 3] Saved: box plots")


# ─────────────────────────────────────────
#   Probability Bar Chart
#           (Conditional Probabilities)
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Graph 4 — Conditional Probability Analysis',
             fontsize=15, fontweight='bold')

prob_features = {
    'Money Terms'     : (P_money_given_legit,   P_money_given_phish),
    'Urgency Terms'   : (P_urgency_given_legit,  P_urgency_given_phish),
    'Suspicious Link' : (P_suslink_given_legit,  P_suslink_given_phish),
    'Has Attachment'  : (legit_df['has_attachment'].mean(),
                         phish_df['has_attachment'].mean()),
}

x_pos   = np.arange(len(prob_features))
width   = 0.35
labels_ = list(prob_features.keys())
legit_p = [v[0] for v in prob_features.values()]
phish_p = [v[1] for v in prob_features.values()]

bars1 = ax.bar(x_pos - width/2, legit_p, width,
               label='Legitimate', color=COLORS['legit'], alpha=0.85, edgecolor='white')
bars2 = ax.bar(x_pos + width/2, phish_p, width,
               label='Phishing',   color=COLORS['phish'], alpha=0.85, edgecolor='white')

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_, fontsize=11)
ax.set_ylabel('Probability', fontsize=11)
ax.set_xlabel('Feature', fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_title('P(Feature | Class) — Higher phishing probability indicates strong signal',
             fontsize=11)

plt.tight_layout()
plt.savefig('graph4_probability_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 4] Saved: probability analysis")


# ─────────────────────────────────────────
#   Correlation Heatmap
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
fig.suptitle('Graph 5 — Feature Correlation Heatmap',
             fontsize=15, fontweight='bold')

corr_df   = df[FEATURES + [LABEL]].copy()
corr_df.columns = [c.replace('_', '\n') for c in corr_df.columns]
corr_matrix = corr_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, mask=mask, ax=ax,
            annot_kws={'size': 8}, linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Correlation between Features and Label', fontsize=12)

plt.tight_layout()
plt.savefig('graph5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 5] Saved: correlation heatmap")


# ─────────────────────────────────────────
#   Confusion Matrix
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle('Graph 6 — Confusion Matrix', fontsize=15, fontweight='bold')

cm_labels = ['Legitimate', 'Phishing']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=cm_labels, yticklabels=cm_labels,
            linewidths=2, linecolor='white',
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label',    fontsize=12)
ax.set_title(f'Naive Bayes — Accuracy: {acc*100:.2f}%', fontsize=12)

# Color annotations
cm_anno = [('TN', 0, 0), ('FP', 0, 1), ('FN', 1, 0), ('TP', 1, 1)]
for label, r, c in cm_anno:
    ax.text(c + 0.5, r + 0.72, label,
            ha='center', va='center', fontsize=10,
            color='gray', style='italic')

plt.tight_layout()
plt.savefig('graph6_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 6] Saved: confusion matrix")


# ─────────────────────────────────────────
#  Performance Metrics Bar Chart
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Graph 7 — Model Performance Metrics', fontsize=15, fontweight='bold')

metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
colors_m = [COLORS['green'], COLORS['legit'], COLORS['accent'], '#9C27B0']

bars = ax.bar(list(metrics.keys()), [v*100 for v in metrics.values()],
              color=colors_m, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val*100:.2f}%',
            ha='center', fontsize=12, fontweight='bold')

ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=11)
ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graph7_performance_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 7] Saved: performance metrics")


# ─────────────────────────────────────────
#   Sender Domain Analysis
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Graph 8 — Sender Domain Analysis', fontsize=15, fontweight='bold')

legit_domains = legit_df['sender_domain'].value_counts().head(5)
phish_domains = phish_df['sender_domain'].value_counts().head(5)

axes[0].barh(legit_domains.index[::-1], legit_domains.values[::-1],
             color=COLORS['legit'], alpha=0.85, edgecolor='white')
axes[0].set_title('Top Domains — Legitimate Emails', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Count', fontsize=10)
for i, v in enumerate(legit_domains.values[::-1]):
    axes[0].text(v + 1, i, str(v), va='center', fontsize=10, fontweight='bold')

axes[1].barh(phish_domains.index[::-1], phish_domains.values[::-1],
             color=COLORS['phish'], alpha=0.85, edgecolor='white')
axes[1].set_title('Top Domains — Phishing Emails', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Count', fontsize=10)
for i, v in enumerate(phish_domains.values[::-1]):
    axes[1].text(v + 1, i, str(v), va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graph8_sender_domains.png', dpi=150, bbox_inches='tight')
plt.close()
print("[GRAPH 8] Saved: sender domain analysis")


# ─────────────────────────────────────────
#  TABLE: Comparison Summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPARISON TABLE — Classification Results Summary")
print("=" * 60)

comparison = pd.DataFrame({
    'Metric'       : ['Total Emails', 'Correctly Classified',
                      'Wrongly Classified', 'Accuracy', 'Precision',
                      'Recall', 'F1-Score', 'False Positive Rate',
                      'False Negative Rate'],
    'Value'        : [len(y_test),
                      int(tp + tn),
                      int(fp + fn),
                      f'{acc*100:.2f}%',
                      f'{prec*100:.2f}%',
                      f'{rec*100:.2f}%',
                      f'{f1*100:.2f}%',
                      f'{fpr*100:.2f}%',
                      f'{fnr*100:.2f}%'],
    'Description'  : ['Test set size',
                      'True Positive + True Negative',
                      'False Positive + False Negative',
                      'Overall correctness',
                      'Of flagged phishing, how many were real',
                      'Of real phishing, how many detected',
                      'Harmonic mean of Precision & Recall',
                      'Legit emails wrongly flagged as phishing',
                      'Phishing emails that slipped through']
})

print(comparison.to_string(index=False))

print("\n" + "=" * 60)
print("  All 8 graphs saved ")
print("  Code complete!")
print("=" * 60)