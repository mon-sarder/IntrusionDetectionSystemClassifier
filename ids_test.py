"""
tests/test_ids.py
Unit + integration tests for the IDS pipeline.
Run:  pytest tests/ -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generate_data import generate_synthetic, load_dataset
from preprocess    import (encode_and_scale, build_binary_label,
                            make_sequences, save_artifacts, load_artifacts)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def raw_df():
    return generate_synthetic(n_total=500, seed=0)


@pytest.fixture(scope='module')
def encoded(raw_df):
    X, scaler, feature_cols = encode_and_scale(raw_df, fit=True)
    y = build_binary_label(raw_df).values
    return X, y, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_shape(self, raw_df):
        assert len(raw_df) == 500

    def test_required_columns(self, raw_df):
        required = ['duration','protocol_type','service','flag',
                    'src_bytes','dst_bytes','label']
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_label_classes(self, raw_df):
        labels = set(raw_df['label'].unique())
        assert labels == {'normal','dos','probe','r2l','u2r'}

    def test_no_nulls(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_numeric_ranges(self, raw_df):
        for col in ['serror_rate','rerror_rate','same_srv_rate']:
            assert raw_df[col].between(0, 1).all(), f"{col} out of [0,1]"

    def test_protocol_values(self, raw_df):
        valid = {'tcp','udp','icmp'}
        assert set(raw_df['protocol_type'].unique()).issubset(valid)

    def test_reproducible(self):
        df1 = generate_synthetic(n_total=100, seed=99)
        df2 = generate_synthetic(n_total=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_synthetic(n_total=100, seed=1)
        df2 = generate_synthetic(n_total=100, seed=2)
        assert not df1['src_bytes'].equals(df2['src_bytes'])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_output_shape(self, encoded):
        X, y, scaler, feature_cols = encoded
        assert X.shape[0] == 500
        assert X.shape[1] == len(feature_cols)
        assert len(y) == 500

    def test_no_nan_after_encode(self, encoded):
        X, *_ = encoded
        assert not np.isnan(X).any()

    def test_scaled_mean_near_zero(self, encoded):
        X, *_ = encoded
        # After StandardScaler, column means should be ≈ 0
        col_means = np.abs(X.mean(axis=0))
        assert col_means.mean() < 0.5   # lenient for small dataset

    def test_feature_cols_no_label(self, encoded):
        _, _, _, feature_cols = encoded
        assert 'label' not in feature_cols
        assert 'binary_label' not in feature_cols

    def test_binary_label_values(self, raw_df):
        y = build_binary_label(raw_df).values
        assert set(y).issubset({0, 1})
        assert y.sum() > 0    # has positives
        assert (y == 0).sum() > 0  # has negatives

    def test_align_on_inference(self, encoded, raw_df):
        """Inference time encode must align to training feature_cols."""
        X_train, _, scaler, feature_cols = encoded
        single = raw_df.iloc[:1].copy()
        X_inf, _, _ = encode_and_scale(single, scaler=scaler,
                                        feature_cols=feature_cols, fit=False)
        assert X_inf.shape[1] == X_train.shape[1]

    def test_unknown_category_handled(self, encoded, raw_df):
        """A record with an unseen service value should still encode without error."""
        _, _, scaler, feature_cols = encoded
        rec = raw_df.iloc[:1].copy()
        rec['service'] = 'totally_new_service_xyz'
        # Should not raise; missing OHE cols filled with 0
        X, _, _ = encode_and_scale(rec, scaler=scaler,
                                    feature_cols=feature_cols, fit=False)
        assert X.shape[1] == len(feature_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence building (LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class TestSequences:

    def test_shape(self, encoded):
        X, y, *_ = encoded
        seq_len = 10
        X_seq, y_seq = make_sequences(X, y, seq_len)
        assert X_seq.shape == (len(X) - seq_len + 1, seq_len, X.shape[1])
        assert len(y_seq) == len(X) - seq_len + 1

    def test_label_at_end(self, encoded):
        """Last label in each window should be y[i + seq_len - 1]."""
        X, y, *_ = encoded
        X_seq, y_seq = make_sequences(X, y, seq_len=5)
        assert y_seq[0] == y[4]
        assert y_seq[1] == y[5]

    def test_seq_len_1(self, encoded):
        X, y, *_ = encoded
        X_seq, y_seq = make_sequences(X, y, seq_len=1)
        assert len(X_seq) == len(X)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifacts:

    def test_save_and_load(self, encoded, tmp_path):
        _, _, scaler, feature_cols = encoded
        save_artifacts(scaler, feature_cols, out_dir=str(tmp_path))
        s2, fc2, le2 = load_artifacts(str(tmp_path))
        assert fc2 == feature_cols
        assert le2 is None   # no label_encoder saved in this call

    def test_transform_consistency(self, encoded, tmp_path):
        X_orig, _, scaler, feature_cols = encoded
        save_artifacts(scaler, feature_cols, out_dir=str(tmp_path))
        s2, fc2, _ = load_artifacts(str(tmp_path))
        # Re-transform a slice and compare
        from generate_data import generate_synthetic
        df_new = generate_synthetic(n_total=10, seed=7)
        X1, _, _ = encode_and_scale(df_new, scaler=scaler, feature_cols=feature_cols, fit=False)
        X2, _, _ = encode_and_scale(df_new, scaler=s2,     feature_cols=fc2,          fit=False)
        np.testing.assert_array_almost_equal(X1, X2, decimal=6)


# ─────────────────────────────────────────────────────────────────────────────
# Flask API
# ─────────────────────────────────────────────────────────────────────────────

class TestFlaskAPI:
    """
    Light smoke-tests against the Flask test client.
    Models are NOT loaded here (no .pkl files in CI temp dir),
    so we test the 503 / error paths and structure-level endpoints.
    """

    @pytest.fixture(scope='class')
    def client(self):
        # Patch MDL_DIR to a non-existent path so models won't load
        import importlib
        import app as app_module
        app_module._ready = False
        app_module._load_error = "No models (CI test mode)"
        app_module.app.config['TESTING'] = True
        with app_module.app.test_client() as c:
            yield c

    def test_status_endpoint(self, client):
        r = client.get('/api/status')
        assert r.status_code == 200
        data = r.get_json()
        assert 'status' in data
        assert 'models' in data

    def test_examples_endpoint(self, client):
        r = client.get('/api/examples')
        assert r.status_code == 200
        data = r.get_json()
        assert 'normal' in data
        assert 'dos' in data
        assert 'probe' in data

    def test_predict_503_when_not_ready(self, client):
        r = client.post('/api/predict',
                        json={'duration': 0, 'protocol_type': 'tcp'},
                        content_type='application/json')
        assert r.status_code == 503

    def test_history_empty(self, client):
        r = client.get('/api/history')
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_stats_empty(self, client):
        r = client.get('/api/stats')
        assert r.status_code == 200
        data = r.get_json()
        assert data['total'] == 0

    def test_batch_not_a_list(self, client):
        r = client.post('/api/predict/batch',
                        json={'not': 'a list'},
                        content_type='application/json')
        # 503 because models not ready, or 400 for bad input
        assert r.status_code in (400, 503)