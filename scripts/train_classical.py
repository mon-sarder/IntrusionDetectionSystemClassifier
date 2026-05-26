# scripts/train_classical.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score,
                              roc_curve)
from sklearn.preprocessing import LabelEncoder

from scripts.generate_data import load_dataset
from scripts.preprocess import (encode_and_scale, build_binary_label,
                                  save_artifacts)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL_DIR = os.path.join(ROOT, 'models')
IMG_DIR = os.path.join(ROOT, 'static', 'img')


def main():
    os.makedirs(MDL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    DATA_DIR = os.path.join(ROOT, 'data')
    df, _, src = load_dataset(DATA_DIR)
    print(f"[train] Data: {src}  |  {len(df):,} rows")

    X, scaler, feature_cols = encode_and_scale(df, fit=True)
    y_binary = build_binary_label(df).values

    le = LabelEncoder()
    y_multi = le.fit_transform(df['label'].str.strip().str.lower())

    save_artifacts(scaler, feature_cols, label_encoder=le, out_dir=MDL_DIR)

    X_tr, X_te, yb_tr, yb_te, ym_tr, ym_te = train_test_split(
        X, y_binary, y_multi, test_size=0.2, random_state=42, stratify=y_binary
    )

    print("[train] Training Random Forest (binary)...")
    rf_b = RandomForestClassifier(n_estimators=100, max_depth=20,
                                   random_state=42, n_jobs=-1)
    rf_b.fit(X_tr, yb_tr)
    joblib.dump(rf_b, os.path.join(MDL_DIR, 'rf_binary.pkl'))

    print("[train] Training Random Forest (multi-class)...")
    rf_m = RandomForestClassifier(n_estimators=100, max_depth=20,
                                   random_state=42, n_jobs=-1)
    rf_m.fit(X_tr, ym_tr)
    joblib.dump(rf_m, os.path.join(MDL_DIR, 'rf_multi.pkl'))

    print("[train] Training SVM (binary)...")
    svm_b = SVC(kernel='rbf', C=1.0, gamma='scale',
                random_state=42, probability=True)
    svm_b.fit(X_tr, yb_tr)
    joblib.dump(svm_b, os.path.join(MDL_DIR, 'svm_binary.pkl'))

    # --- Evaluation ---
    yb_pred_rf  = rf_b.predict(X_te)
    yb_pred_svm = svm_b.predict(X_te)
    ym_pred_rf  = rf_m.predict(X_te)

    print("\n[train] RF Binary:", classification_report(yb_te, yb_pred_rf,
          target_names=['Normal','Attack']))

    # --- ROC Curve ---
    fpr_rf,  tpr_rf,  _ = roc_curve(yb_te, rf_b.predict_proba(X_te)[:,1])
    fpr_svm, tpr_svm, _ = roc_curve(yb_te, svm_b.predict_proba(X_te)[:,1])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(fpr_rf,  tpr_rf,  label=f'RF  AUC={roc_auc_score(yb_te, rf_b.predict_proba(X_te)[:,1]):.4f}')
    ax.plot(fpr_svm, tpr_svm, label=f'SVM AUC={roc_auc_score(yb_te, svm_b.predict_proba(X_te)[:,1]):.4f}')
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'roc_curve.png'), dpi=120)
    plt.close()

    # --- Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, y_pred, title, cmap in zip(
        axes,
        [yb_pred_rf, yb_pred_svm],
        ['RF Binary', 'SVM Binary'],
        ['Blues', 'Purples']
    ):
        sns.heatmap(confusion_matrix(yb_te, y_pred), annot=True, fmt='d',
                    cmap=cmap, ax=ax, xticklabels=['Normal','Attack'],
                    yticklabels=['Normal','Attack'])
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'confusion_matrices.png'), dpi=120)
    plt.close()

    # --- Feature Importance ---
    importances = rf_b.feature_importances_
    indices = np.argsort(importances)[::-1][:25]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(25), importances[indices][::-1])
    ax.set_yticks(range(25))
    ax.set_yticklabels([feature_cols[i] for i in indices][::-1], fontsize=8)
    ax.set_title('Top 25 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'feature_importance.png'), dpi=120)
    plt.close()

    # --- CV Scores ---
    cv_scores = cross_val_score(rf_b, X, y_binary, cv=5, scoring='f1')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, 6), cv_scores)
    ax.axhline(cv_scores.mean(), color='red', linestyle='--',
               label=f'Mean={cv_scores.mean():.4f}')
    ax.set_title('5-Fold CV F1 Scores'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'cv_scores.png'), dpi=120)
    plt.close()

    print("\n[train] ✓ All models saved to models/")
    print(f"         RF  binary acc: {accuracy_score(yb_te, yb_pred_rf)*100:.2f}%")
    print(f"         SVM binary acc: {accuracy_score(yb_te, yb_pred_svm)*100:.2f}%")


if __name__ == '__main__':
    main()