"""
preprocess.py
Shared preprocessing logic used by training, API, and LSTM pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os

CAT_COLS = ['protocol_type', 'service', 'flag']
DROP_COLS = ['attack_category', 'difficulty_level']  # synthetic extras


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop helper columns if present, strip label whitespace."""
    df = df.copy()
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    df['label'] = df['label'].str.strip().str.lower()
    # Map all non-normal to 'attack' for binary; keep raw for multi
    return df


def build_binary_label(df: pd.DataFrame) -> pd.Series:
    return (df['label'] != 'normal').astype(int)


def encode_and_scale(df: pd.DataFrame, scaler=None, feature_cols=None, fit=True):
    """
    One-hot encode categoricals, scale numerics.
    Returns (X_scaled, scaler, feature_cols)
    """
    df = _clean(df)
    df_enc = pd.get_dummies(df, columns=CAT_COLS)
    drop = [c for c in ['label'] if c in df_enc.columns]
    df_enc = df_enc.drop(columns=drop)

    if feature_cols is not None:
        # Align to training columns (adds missing, removes extra)
        for col in feature_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[feature_cols]
    else:
        feature_cols = list(df_enc.columns)

    X = df_enc.astype(float).values

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, scaler, feature_cols


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10):
    """
    Slide a window over X/y to build sequences for LSTM.
    Returns (X_seq, y_seq) with shape (N-seq_len+1, seq_len, features).
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def save_artifacts(scaler, feature_cols, label_encoder=None, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler,       os.path.join(out_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(out_dir, 'feature_cols.pkl'))
    if label_encoder:
        joblib.dump(label_encoder, os.path.join(out_dir, 'label_encoder.pkl'))
    print(f"[preprocess] Artifacts saved to {out_dir}/")


def load_artifacts(out_dir='models'):
    scaler       = joblib.load(os.path.join(out_dir, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(out_dir, 'feature_cols.pkl'))
    le_path = os.path.join(out_dir, 'label_encoder.pkl')
    label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None
    return scaler, feature_cols, label_encoder