"""
train_classical.py
Trains Random Forest and SVM classifiers on NSL-KDD data.
Produces:
  - models/rf_binary.pkl
  - models/rf_multi.pkl
  - models/svm_binary.pkl
  - models/scaler.pkl  + feature_cols.pkl  + label_encoder.pkl
  - static/img/roc_curve.png
  - static/img/confusion_matrices.png
  - static/img/feature_importance.png
  - static/img/cv_scores.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    roc_curve, auc, RocCurveDisplay
)

from generate_data import load_dataset
from preprocess    import (encode_and_scale, build_binary_label,
                            save_artifacts, CAT_COLS)

# ── Style ────────────────────────────────────────────────────────────────────
DARK    = '#0d1117'
PANEL   = '#161b22'
ACCENT  = '#58a6ff'
GREEN   = '#3fb950'
RED     = '#f85149'
YELLOW  = '#d29922'
PURPLE  = '#bc8cff'
TEXT    = '#e6edf3'
SUBTEXT = '#8b949e'
BORDER  = '#30363d'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': SUBTEXT, 'ytick.color': SUBTEXT,
    'axes.edgecolor': BORDER, 'axes.grid': True,
    'grid.color': BORDER, 'grid.linewidth': 0.5,
    'font.family': 'monospace', 'font.size': 10,
})

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'img')
MDL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)


def style_ax(ax, title, fontsize=11):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=ACCENT, fontsize=fontsize, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  IDS Classical ML Training Pipeline")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_df, test_df, src = load_dataset(DATA_DIR)
    print(f"\n[1] Data source: {src}  |  rows: {len(train_df):,}")
    print(train_df['label'].value_counts().to_string())

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2] Preprocessing …")
    X_scaled, scaler, feature_cols = encode_and_scale(train_df, fit=True)
    y_binary = build_binary_label(train_df).values

    le = LabelEncoder()
    y_multi = le.fit_transform(train_df['label'].str.strip().str.lower())

    save_artifacts(scaler, feature_cols, le, MDL_DIR)
    print(f"    Features: {len(feature_cols)}  |  Positives: {y_binary.sum():,} ({y_binary.mean()*100:.1f}%)")

    # ── 3. Train / test split ────────────────────────────────────────────────
    (X_tr, X_te,
     yb_tr, yb_te,
     ym_tr, ym_te) = train_test_split(
        X_scaled, y_binary, y_multi,
        test_size=0.20, random_state=42, stratify=y_binary
    )
    print(f"    Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("\n[3] Training …")

    rf_b = RandomForestClassifier(n_estimators=150, max_depth=25,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_b.fit(X_tr, yb_tr)
    yb_pred_rf  = rf_b.predict(X_te)
    yb_proba_rf = rf_b.predict_proba(X_te)[:, 1]
    print(f"    RF  (binary)  acc={accuracy_score(yb_te, yb_pred_rf)*100:.2f}%  "
          f"f1={f1_score(yb_te, yb_pred_rf)*100:.2f}%")

    rf_m = RandomForestClassifier(n_estimators=150, max_depth=25,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_m.fit(X_tr, ym_tr)
    ym_pred_rf = rf_m.predict(X_te)
    print(f"    RF  (multi)   acc={accuracy_score(ym_te, ym_pred_rf)*100:.2f}%  "
          f"f1={f1_score(ym_te, ym_pred_rf, average='weighted')*100:.2f}%")

    svm_b = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)
    svm_b.fit(X_tr, yb_tr)
    yb_pred_svm  = svm_b.predict(X_te)
    yb_proba_svm = svm_b.predict_proba(X_te)[:, 1]
    print(f"    SVM (binary)  acc={accuracy_score(yb_te, yb_pred_svm)*100:.2f}%  "
          f"f1={f1_score(yb_te, yb_pred_svm)*100:.2f}%")

    # Save models
    joblib.dump(rf_b,  os.path.join(MDL_DIR, 'rf_binary.pkl'))
    joblib.dump(rf_m,  os.path.join(MDL_DIR, 'rf_multi.pkl'))
    joblib.dump(svm_b, os.path.join(MDL_DIR, 'svm_binary.pkl'))
    print(f"    Models saved → {MDL_DIR}/")

    # ── 5. Cross-validation ───────────────────────────────────────────────────
    print("\n[4] Cross-validation (5-fold, RF binary) …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_b, X_scaled, y_binary, cv=cv,
                                scoring='f1', n_jobs=-1)
    print(f"    F1 per fold: {np.round(cv_scores*100, 2)}")
    print(f"    Mean F1: {cv_scores.mean()*100:.3f}% ± {cv_scores.std()*100:.3f}%")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\n[5] Generating plots …")

    # ── 6a. ROC curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK)
    style_ax(ax, 'ROC Curves — Binary Classifiers', 13)

    for proba, name, color in [
        (yb_proba_rf,  'Random Forest', ACCENT),
        (yb_proba_svm, 'SVM (RBF)',     PURPLE),
    ]:
        fpr, tpr, _ = roc_curve(yb_te, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name}  (AUC = {roc_auc:.4f})')

    ax.plot([0,1],[0,1], color=SUBTEXT, lw=1.5, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.05, color=ACCENT)
    ax.set_xlabel('False Positive Rate', color=SUBTEXT)
    ax.set_ylabel('True Positive Rate', color=SUBTEXT)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'roc_curve.png'), dpi=140,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("    Saved: roc_curve.png")

    # ── 6b. Confusion matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Confusion Matrices', color=TEXT, fontsize=14, fontweight='bold')

    configs = [
        (confusion_matrix(yb_te, yb_pred_rf),  ['Normal','Attack'], 'Blues',  'RF Binary'),
        (confusion_matrix(yb_te, yb_pred_svm), ['Normal','Attack'], 'Purples','SVM Binary'),
        (confusion_matrix(ym_te, ym_pred_rf),  le.classes_,         'YlOrRd', 'RF Multi-class'),
    ]
    for ax, (cm, labels, cmap, title) in zip(axes, configs):
        style_ax(ax, title)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor=BORDER,
                    annot_kws={'color': TEXT, 'fontsize': 11, 'fontweight': 'bold'},
                    cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted', color=SUBTEXT)
        ax.set_ylabel('Actual',    color=SUBTEXT)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'confusion_matrices.png'), dpi=140,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("    Saved: confusion_matrices.png")

    # ── 6c. Feature importance ────────────────────────────────────────────────
    importances = pd.Series(rf_b.feature_importances_, index=feature_cols).nlargest(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK)
    style_ax(ax, 'Top 25 Feature Importances — Random Forest', 13)
    colors = [ACCENT]*5 + [GREEN]*5 + [YELLOW]*5 + [PURPLE]*10
    ax.barh(range(25), importances.values[::-1], color=colors[::-1],
            edgecolor=BORDER, linewidth=0.6)
    ax.set_yticks(range(25))
    ax.set_yticklabels([n.replace('_',' ') for n in importances.index[::-1]], fontsize=9)
    ax.set_xlabel('Importance', color=SUBTEXT)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'feature_importance.png'), dpi=140,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("    Saved: feature_importance.png")

    # ── 6d. Cross-validation scores ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(DARK)
    style_ax(ax, '5-Fold Cross-Validation F1 Scores — RF Binary', 12)
    folds = [f'Fold {i+1}' for i in range(5)]
    bars = ax.bar(folds, cv_scores * 100, color=ACCENT, edgecolor=BORDER, linewidth=0.8)
    ax.axhline(cv_scores.mean()*100, color=GREEN, lw=2, linestyle='--',
               label=f'Mean = {cv_scores.mean()*100:.3f}%')
    ax.fill_between(range(5),
                    (cv_scores.mean() - cv_scores.std())*100,
                    (cv_scores.mean() + cv_scores.std())*100,
                    alpha=0.15, color=GREEN)
    ax.set_ylim(max(0, cv_scores.min()*100 - 2), 101)
    ax.set_ylabel('F1 Score (%)', color=SUBTEXT)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    for bar, v in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width()/2, v*100 + 0.1,
                f'{v*100:.3f}%', ha='center', va='bottom', color=TEXT, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'cv_scores.png'), dpi=140,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("    Saved: cv_scores.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    rows = [
        ['RF Binary',       f"{accuracy_score(yb_te,yb_pred_rf)*100:.2f}%",
                            f"{f1_score(yb_te,yb_pred_rf)*100:.2f}%",
                            f"{auc(*roc_curve(yb_te,yb_proba_rf)[:2]):.4f}"],
        ['RF Multi-class',  f"{accuracy_score(ym_te,ym_pred_rf)*100:.2f}%",
                            f"{f1_score(ym_te,ym_pred_rf,average='weighted')*100:.2f}%", '—'],
        ['SVM Binary',      f"{accuracy_score(yb_te,yb_pred_svm)*100:.2f}%",
                            f"{f1_score(yb_te,yb_pred_svm)*100:.2f}%",
                            f"{auc(*roc_curve(yb_te,yb_proba_svm)[:2]):.4f}"],
        ['RF CV (mean)',    f"{cv_scores.mean()*100:.3f}%",
                            f"±{cv_scores.std()*100:.3f}%", '5-fold'],
    ]
    header = f"{'Model':<20} {'Accuracy':>12} {'F1':>10} {'AUC/CV':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r[0]:<20} {r[1]:>12} {r[2]:>10} {r[3]:>10}")
    print("\n✓ Training complete.")

    return rf_b, rf_m, svm_b, scaler, feature_cols, le


if __name__ == '__main__':
    main()