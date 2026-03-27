"""
Simple Intrusion Detection System (IDS)
Based on NSL-KDD Dataset structure
Author: Mon Sarder | Cal Poly Pomona CS + Cybersecurity
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("   INTRUSION DETECTION SYSTEM — NSL-KDD Pipeline")
print("=" * 60)

df = pd.read_csv('nsl_kdd_synthetic.csv')
print(f"\n[1] Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
print(f"\nClass distribution:")
print(df['label'].value_counts().to_string())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2] Preprocessing...")

# Drop the attack_category (human-readable) and difficulty_level columns
df = df.drop(columns=['attack_category', 'difficulty_level'])

# Binary label: normal=0, attack=1
df['binary_label'] = (df['label'] != 'normal').astype(int)

# Identify categorical features
cat_cols = ['protocol_type', 'service', 'flag']
num_cols = [c for c in df.columns if c not in cat_cols + ['label', 'binary_label']]

# One-Hot Encode categoricals
df_encoded = pd.get_dummies(df, columns=cat_cols)

# Drop original label columns, set X and y
feature_cols = [c for c in df_encoded.columns if c not in ['label', 'binary_label']]
X = df_encoded[feature_cols].astype(float)
y_binary = df_encoded['binary_label']
y_multi  = df['label']

# Encode multi-class labels
le = LabelEncoder()
y_multi_enc = le.fit_transform(y_multi)

print(f"   Features after encoding: {X.shape[1]}")
print(f"   Positive (attack) samples: {y_binary.sum():,} ({y_binary.mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
    X, y_binary, y_multi_enc,
    test_size=0.2, random_state=42, stratify=y_binary
)

# Scale numerical features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[3] Training models...")

# --- Random Forest (Binary) ---
rf_binary = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_binary.fit(X_train, yb_train)
yb_pred_rf = rf_binary.predict(X_test)
print(f"   Random Forest (binary)  — Accuracy: {accuracy_score(yb_test, yb_pred_rf)*100:.2f}%")

# --- Random Forest (Multi-class) ---
rf_multi = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_multi.fit(X_train, ym_train)
ym_pred_rf = rf_multi.predict(X_test)
print(f"   Random Forest (multi)   — Accuracy: {accuracy_score(ym_test, ym_pred_rf)*100:.2f}%")

# --- SVM (Binary, scaled) ---
svm_binary = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_binary.fit(X_train_scaled, yb_train)
yb_pred_svm = svm_binary.predict(X_test_scaled)
print(f"   SVM RBF (binary)        — Accuracy: {accuracy_score(yb_test, yb_pred_svm)*100:.2f}%")

# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
print("\n[4] Evaluation — Random Forest (Binary)")
print("-" * 40)
print(classification_report(yb_test, yb_pred_rf, target_names=['Normal', 'Attack']))

print("[4] Evaluation — Random Forest (Multi-class Attack Types)")
print("-" * 40)
print(classification_report(ym_test, ym_pred_rf, target_names=le.classes_))

print("[4] Evaluation — SVM (Binary)")
print("-" * 40)
print(classification_report(yb_test, yb_pred_svm, target_names=['Normal', 'Attack']))

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
print("\n[5] Generating report figures...")

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK_BG   = '#0d1117'
PANEL_BG  = '#161b22'
ACCENT    = '#58a6ff'
RED       = '#f85149'
GREEN     = '#3fb950'
YELLOW    = '#d29922'
TEXT      = '#e6edf3'
SUBTEXT   = '#8b949e'
BORDER    = '#30363d'

plt.rcParams.update({
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': SUBTEXT, 'ytick.color': SUBTEXT,
    'axes.edgecolor': BORDER, 'font.family': 'monospace',
})

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

# ── Plot 1: Class Distribution ──
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Dataset Class Distribution')
counts = df['label'].value_counts()
colors = [GREEN, RED, YELLOW, '#f0883e', '#bc8cff']
bars = ax1.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor=BORDER, linewidth=0.8)
ax1.set_xlabel('Attack Type', color=SUBTEXT)
ax1.set_ylabel('Count', color=SUBTEXT)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val:,}', ha='center', va='bottom', color=TEXT, fontsize=9)

# ── Plot 2: Confusion Matrix RF Binary ──
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'RF Confusion Matrix (Binary)')
cm = confusion_matrix(yb_test, yb_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
            linewidths=0.5, linecolor=BORDER,
            annot_kws={'color': TEXT, 'fontsize': 12, 'fontweight': 'bold'})
ax2.set_xlabel('Predicted', color=SUBTEXT)
ax2.set_ylabel('Actual', color=SUBTEXT)

# ── Plot 3: Confusion Matrix SVM Binary ──
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'SVM Confusion Matrix (Binary)')
cm_svm = confusion_matrix(yb_test, yb_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', ax=ax3,
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
            linewidths=0.5, linecolor=BORDER,
            annot_kws={'color': TEXT, 'fontsize': 12, 'fontweight': 'bold'})
ax3.set_xlabel('Predicted', color=SUBTEXT)
ax3.set_ylabel('Actual', color=SUBTEXT)

# ── Plot 4: Multi-class Confusion Matrix ──
ax4 = fig.add_subplot(gs[1, :2])
style_ax(ax4, 'RF Confusion Matrix (Multi-class Attack Types)')
cm_multi = confusion_matrix(ym_test, ym_pred_rf)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='YlOrRd', ax=ax4,
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, linecolor=BORDER,
            annot_kws={'color': '#0d1117', 'fontsize': 11, 'fontweight': 'bold'})
ax4.set_xlabel('Predicted', color=SUBTEXT)
ax4.set_ylabel('Actual', color=SUBTEXT)

# ── Plot 5: Model Comparison ──
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, 'Model Accuracy Comparison')
models = ['RF\n(Binary)', 'RF\n(Multi)', 'SVM\n(Binary)']
accs = [
    accuracy_score(yb_test, yb_pred_rf)*100,
    accuracy_score(ym_test, ym_pred_rf)*100,
    accuracy_score(yb_test, yb_pred_svm)*100,
]
f1s = [
    f1_score(yb_test, yb_pred_rf)*100,
    f1_score(ym_test, ym_pred_rf, average='weighted')*100,
    f1_score(yb_test, yb_pred_svm)*100,
]
x = np.arange(len(models))
w = 0.35
b1 = ax5.bar(x - w/2, accs, w, label='Accuracy', color=ACCENT, alpha=0.85, edgecolor=BORDER)
b2 = ax5.bar(x + w/2, f1s, w, label='F1 Score', color=GREEN, alpha=0.85, edgecolor=BORDER)
ax5.set_xticks(x)
ax5.set_xticklabels(models, color=TEXT)
ax5.set_ylim(0, 115)
ax5.set_ylabel('Score (%)', color=SUBTEXT)
ax5.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
for b, v in zip(list(b1) + list(b2), accs + f1s):
    ax5.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:.1f}%',
             ha='center', va='bottom', color=TEXT, fontsize=8)

# ── Plot 6: Feature Importances ──
ax6 = fig.add_subplot(gs[2, :])
style_ax(ax6, 'Top 20 Feature Importances (Random Forest)')
importances = pd.Series(rf_binary.feature_importances_, index=X.columns)
top20 = importances.nlargest(20)
colors_feat = [ACCENT if i < 5 else (GREEN if i < 10 else YELLOW) for i in range(20)]
bars_feat = ax6.barh(range(20), top20.values[::-1], color=colors_feat[::-1],
                     edgecolor=BORDER, linewidth=0.5)
ax6.set_yticks(range(20))
ax6.set_yticklabels([n.replace('_', ' ') for n in top20.index[::-1]], fontsize=9, color=TEXT)
ax6.set_xlabel('Importance Score', color=SUBTEXT)
ax6.invert_xaxis()
ax6.yaxis.tick_right()
for bar, val in zip(bars_feat, top20.values[::-1]):
    ax6.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left', color=SUBTEXT, fontsize=8)

# Title
fig.suptitle('Intrusion Detection System — NSL-KDD Analysis',
             color=TEXT, fontsize=16, fontweight='bold', y=0.98)

plt.savefig('ids_report.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG, edgecolor='none')
print("   Saved: ids_report.png")

# ─────────────────────────────────────────────
# 7. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("   FINAL RESULTS SUMMARY")
print("=" * 60)
results = {
    'Model': ['RF Binary', 'RF Multi-class', 'SVM Binary'],
    'Accuracy': [f'{accuracy_score(yb_test, yb_pred_rf)*100:.2f}%',
                 f'{accuracy_score(ym_test, ym_pred_rf)*100:.2f}%',
                 f'{accuracy_score(yb_test, yb_pred_svm)*100:.2f}%'],
    'F1 Score': [f'{f1_score(yb_test, yb_pred_rf)*100:.2f}%',
                 f'{f1_score(ym_test, ym_pred_rf, average="weighted")*100:.2f}%',
                 f'{f1_score(yb_test, yb_pred_svm)*100:.2f}%'],
}
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print("\n✓ Pipeline complete.")