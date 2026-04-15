"""
train_lstm.py
Trains an LSTM for sequence-based intrusion detection.

Architecture:
  Input (seq_len=10, n_features)
  → LSTM(128, return_sequences=True)
  → Dropout(0.3)
  → LSTM(64)
  → Dropout(0.3)
  → Dense(64, relu) → BatchNorm
  → Dense(1, sigmoid)   ← binary
       OR
  → Dense(n_classes, softmax) ← multi-class

Run:  python train_lstm.py [--seq-len 10] [--epochs 30] [--mode binary|multi]

Requirements:
  pip install tensorflow pandas scikit-learn matplotlib joblib
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Try importing TensorFlow; give clear error if absent ──────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                          BatchNormalization, Input)
    from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                             ModelCheckpoint)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from scripts.generate_data import load_dataset
from scripts.preprocess import (encode_and_scale, build_binary_label,
                                make_sequences, load_artifacts)

DARK   = '#0d1117'; PANEL  = '#161b22'; ACCENT = '#58a6ff'
GREEN  = '#3fb950'; RED    = '#f85149'; TEXT   = '#e6edf3'
SUBTEXT= '#8b949e'; BORDER = '#30363d'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': SUBTEXT, 'ytick.color': SUBTEXT,
    'axes.edgecolor': BORDER, 'axes.grid': True,
    'grid.color': BORDER, 'grid.linewidth': 0.5,
    'font.family': 'monospace',
})

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'img')
MDL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def build_lstm_binary(seq_len, n_features):
    """Two-layer stacked LSTM for binary classification."""
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True, name='lstm_1'),
        Dropout(0.30, name='drop_1'),
        LSTM(64, return_sequences=False, name='lstm_2'),
        Dropout(0.30, name='drop_2'),
        Dense(64, activation='relu', name='fc_1'),
        BatchNormalization(name='bn_1'),
        Dense(1, activation='sigmoid', name='output'),
    ], name='IDS_LSTM_Binary')
    model.compile(optimizer=Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_lstm_multi(seq_len, n_features, n_classes):
    """Two-layer stacked LSTM for multi-class classification."""
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True, name='lstm_1'),
        Dropout(0.30, name='drop_1'),
        LSTM(64, return_sequences=False, name='lstm_2'),
        Dropout(0.30, name='drop_2'),
        Dense(128, activation='relu', name='fc_1'),
        BatchNormalization(name='bn_1'),
        Dense(n_classes, activation='softmax', name='output'),
    ], name='IDS_LSTM_Multi')
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history(history, mode, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(f'LSTM Training History ({mode})', color=TEXT,
                 fontsize=14, fontweight='bold')

    for ax, metric, title in zip(axes,
                                  ['loss', 'accuracy'],
                                  ['Loss', 'Accuracy']):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=ACCENT, fontsize=11, fontweight='bold')
        ax.plot(history.history[metric],       color=ACCENT,  lw=2, label='Train')
        ax.plot(history.history[f'val_{metric}'], color=GREEN, lw=2, label='Val')
        ax.set_xlabel('Epoch', color=SUBTEXT)
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor=DARK)
    plt.close()


def plot_cm(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.5),
                                     max(5, len(labels)*1.2)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=ACCENT, fontsize=12, fontweight='bold')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor=BORDER,
                annot_kws={'color': TEXT, 'fontsize': 11, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted', color=SUBTEXT)
    ax.set_ylabel('Actual',    color=SUBTEXT)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor=DARK)
    plt.close()


def main(seq_len=10, epochs=30, mode='binary', batch_size=256):
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow not installed.")
        print("  pip install tensorflow")
        sys.exit(1)

    print("=" * 65)
    print(f"  IDS LSTM Training Pipeline  |  mode={mode}  seq_len={seq_len}")
    print("=" * 65)

    # ── Load + preprocess ────────────────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    df, _, src = load_dataset(DATA_DIR)
    print(f"\n[1] Data: {src}  |  {len(df):,} rows")

    # Load scaler/feature_cols built by train_classical.py
    try:
        scaler, feature_cols, le = load_artifacts(MDL_DIR)
        X_scaled, _, _ = encode_and_scale(df, scaler=scaler,
                                           feature_cols=feature_cols, fit=False)
        print(f"[1] Loaded existing scaler/feature_cols from {MDL_DIR}/")
    except Exception:
        print("[1] No existing artifacts — fitting scaler now")
        X_scaled, scaler, feature_cols = encode_and_scale(df, fit=True)
        le = None

    y_binary = build_binary_label(df).values
    le_multi = LabelEncoder()
    y_multi  = le_multi.fit_transform(df['label'].str.strip().str.lower())
    n_classes = len(le_multi.classes_)

    # ── Build sequences ───────────────────────────────────────────────────────
    print(f"\n[2] Building sequences (len={seq_len}) …")
    if mode == 'binary':
        X_seq, y_seq = make_sequences(X_scaled, y_binary, seq_len)
    else:
        X_seq, y_seq = make_sequences(X_scaled, y_multi, seq_len)

    print(f"    X_seq: {X_seq.shape}  y_seq: {y_seq.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y_seq, test_size=0.20, random_state=42,
        stratify=(y_seq if mode == 'binary' else None)
    )

    if mode == 'multi':
        y_tr_cat = to_categorical(y_tr, n_classes)
        y_te_cat = to_categorical(y_te, n_classes)
    else:
        y_tr_cat, y_te_cat = y_tr.astype(float), y_te.astype(float)

    n_features = X_seq.shape[2]

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\n[3] Building LSTM ({mode}) …")
    if mode == 'binary':
        model = build_lstm_binary(seq_len, n_features)
    else:
        model = build_lstm_multi(seq_len, n_features, n_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(MDL_DIR, f'lstm_{mode}_best.keras')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[4] Training …  (epochs={epochs}, batch={batch_size})")
    history = model.fit(
        X_tr, y_tr_cat,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5] Evaluation …")
    if mode == 'binary':
        y_pred_prob = model.predict(X_te, verbose=0).flatten()
        y_pred      = (y_pred_prob >= 0.5).astype(int)
        print(classification_report(y_te, y_pred, target_names=['Normal', 'Attack']))
        cm_labels = ['Normal', 'Attack']
    else:
        y_pred_prob = model.predict(X_te, verbose=0)
        y_pred      = np.argmax(y_pred_prob, axis=1)
        print(classification_report(y_te, y_pred, target_names=le_multi.classes_))
        cm_labels = list(le_multi.classes_)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(IMG_DIR, exist_ok=True)
    plot_history(history, mode,
                 os.path.join(IMG_DIR, f'lstm_history_{mode}.png'))
    print(f"    Saved: lstm_history_{mode}.png")

    plot_cm(y_te, y_pred, cm_labels,
            f'LSTM Confusion Matrix ({mode})',
            os.path.join(IMG_DIR, f'lstm_cm_{mode}.png'))
    print(f"    Saved: lstm_cm_{mode}.png")

    # Save final model
    final_path = os.path.join(MDL_DIR, f'lstm_{mode}_final.keras')
    model.save(final_path)
    print(f"    Model saved → {final_path}")
    print("\n✓ LSTM training complete.")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len',    type=int, default=10)
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--mode',       choices=['binary','multi'], default='binary')
    args = parser.parse_args()
    main(seq_len=args.seq_len, epochs=args.epochs,
         mode=args.mode, batch_size=args.batch_size)