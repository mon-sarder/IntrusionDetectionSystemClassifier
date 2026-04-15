"""
app.py — IDS Flask API + Dashboard
All scripts live at repo root (flat structure).

Endpoints:
  GET  /                    → Dashboard HTML
  GET  /api/status          → Model + system info
  POST /api/predict         → Single-record prediction
  POST /api/predict/batch   → Batch prediction (≤500)
  GET  /api/history         → Last N predictions
  GET  /api/stats           → Aggregated stats for dashboard charts
  POST /api/retrain         → Trigger model retraining (async)
  GET  /api/examples        → Example NSL-KDD records
"""

import os, sys, time, threading, uuid
from datetime import datetime
from collections import deque, defaultdict

# ── Flat-structure path fix ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder=os.path.join(ROOT, 'templates'))
app.config['JSON_SORT_KEYS'] = False

MDL_DIR = os.path.join(ROOT, 'models')
IMG_DIR = os.path.join(ROOT, 'static', 'img')

# ── Lazy model loading ────────────────────────────────────────────────────────
_models     = {}
_ready      = False
_load_error = None


def _load_models():
    global _models, _ready, _load_error
    try:
        import joblib
        from scripts.preprocess import load_artifacts

        scaler, feature_cols, le = load_artifacts(MDL_DIR)
        rf_b  = joblib.load(os.path.join(MDL_DIR, 'rf_binary.pkl'))
        rf_m  = joblib.load(os.path.join(MDL_DIR, 'rf_multi.pkl'))
        svm_b = joblib.load(os.path.join(MDL_DIR, 'svm_binary.pkl'))

        lstm = None
        try:
            import tensorflow as tf
            p = os.path.join(MDL_DIR, 'lstm_binary_final.keras')
            if os.path.exists(p):
                lstm = tf.keras.models.load_model(p)
        except Exception:
            pass

        _models = {
            'rf_binary':     rf_b,
            'rf_multi':      rf_m,
            'svm_binary':    svm_b,
            'lstm':          lstm,
            'scaler':        scaler,
            'feature_cols':  feature_cols,
            'label_encoder': le,
        }
        _ready = True
        print("[app] ✓ Models loaded")
    except Exception as e:
        _load_error = str(e)
        print(f"[app] ✗ Model loading failed: {e}")
        print("[app]   Run  python train_classical.py  first")


threading.Thread(target=_load_models, daemon=True).start()

# ── In-memory history ─────────────────────────────────────────────────────────
MAX_HISTORY = 1000
_history = deque(maxlen=MAX_HISTORY)
_stats   = defaultdict(int)
_app_start = time.time()


# ── Core prediction helper ────────────────────────────────────────────────────
def _predict_one(record: dict, model_name: str = 'rf_binary') -> dict:
    import pandas as pd
    from scripts.preprocess import encode_and_scale

    record = dict(record)
    record.setdefault('label', 'normal')

    df = pd.DataFrame([record])
    X, _, _ = encode_and_scale(
        df,
        scaler=_models['scaler'],
        feature_cols=_models['feature_cols'],
        fit=False,
    )

    model     = _models.get(model_name) or _models['rf_binary']
    pred      = model.predict(X)[0]
    proba     = model.predict_proba(X)[0]
    binary    = 'ATTACK' if pred == 1 else 'NORMAL'
    confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])

    multi_pred  = _models['rf_multi'].predict(X)[0]
    le          = _models['label_encoder']
    attack_type = le.inverse_transform([multi_pred])[0].upper()

    return {
        'id':          str(uuid.uuid4())[:8],
        'timestamp':   datetime.now().isoformat(),
        'prediction':  binary,
        'attack_type': attack_type if binary == 'ATTACK' else 'N/A',
        'confidence':  round(confidence, 4),
        'model_used':  model_name,
        'features_in': len(_models['feature_cols']),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/status')
def status():
    return jsonify({
        'status':  'ready' if _ready else 'loading',
        'error':   _load_error,
        'models': {
            'rf_binary':  _ready,
            'rf_multi':   _ready,
            'svm_binary': _ready,
            'lstm':       _models.get('lstm') is not None,
        },
        'total_predictions': sum(_stats.values()),
        'uptime_s':          round(time.time() - _app_start, 1),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if not _ready:
        return jsonify({'error': 'Models not loaded — run train_classical.py first'}), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    model_name = data.pop('model', 'rf_binary')
    if model_name not in ('rf_binary', 'svm_binary'):
        model_name = 'rf_binary'

    try:
        result = _predict_one(data, model_name)
        _history.appendleft(result)
        _stats[result['prediction']] += 1
        _stats[result['attack_type']] += 1
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    if not _ready:
        return jsonify({'error': 'Models not loaded'}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a JSON array'}), 400
    if len(data) > 500:
        return jsonify({'error': 'Max batch size is 500'}), 400

    results = []
    for record in data:
        mn = record.pop('model', 'rf_binary')
        try:
            r = _predict_one(record, mn)
            _history.appendleft(r)
            _stats[r['prediction']] += 1
            _stats[r['attack_type']] += 1
            results.append(r)
        except Exception as e:
            results.append({'error': str(e)})

    return jsonify({'count': len(results), 'results': results})


@app.route('/api/history')
def history():
    n = min(int(request.args.get('n', 50)), MAX_HISTORY)
    return jsonify(list(_history)[:n])


@app.route('/api/stats')
def stats():
    h       = list(_history)
    total   = len(h)
    attacks = sum(1 for r in h if r.get('prediction') == 'ATTACK')
    by_type = defaultdict(int)
    conf_sum = 0.0
    for r in h:
        if r.get('prediction') == 'ATTACK':
            by_type[r.get('attack_type', 'UNKNOWN')] += 1
        conf_sum += r.get('confidence', 0)

    return jsonify({
        'total':          total,
        'attacks':        attacks,
        'normals':        total - attacks,
        'attack_rate':    round(attacks / total, 4) if total else 0,
        'avg_confidence': round(conf_sum / total, 4) if total else 0,
        'by_attack_type': dict(by_type),
        'images': {
            'roc':       '/static/img/roc_curve.png',
            'cm':        '/static/img/confusion_matrices.png',
            'fi':        '/static/img/feature_importance.png',
            'cv':        '/static/img/cv_scores.png',
            'lstm_hist': '/static/img/lstm_history_binary.png',
            'lstm_cm':   '/static/img/lstm_cm_binary.png',
        },
    })


@app.route('/api/retrain', methods=['POST'])
def retrain():
    if not _ready:
        return jsonify({'error': 'Initial training not done yet'}), 503

    def _do():
        try:
            from scripts.train_classical import main as train_main
            train_main()
            _load_models()
        except Exception as e:
            print(f"[retrain] Error: {e}")

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({'status': 'retraining started in background'})


@app.route('/static/img/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)


EXAMPLE_RECORDS = {
    'normal': {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 181, "dst_bytes": 5450, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 8, "srv_count": 8, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 9,
        "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    },
    'dos': {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 1.0, "srv_serror_rate": 1.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    },
    'probe': {
        "duration": 0, "protocol_type": "tcp", "service": "other", "flag": "REJ",
        "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 41, "srv_count": 6, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 1.0, "srv_rerror_rate": 1.0, "same_srv_rate": 0.15,
        "diff_srv_rate": 0.85, "srv_diff_host_rate": 1.0, "dst_host_count": 255,
        "dst_host_srv_count": 6, "dst_host_same_srv_rate": 0.02,
        "dst_host_diff_srv_rate": 0.98, "dst_host_same_src_port_rate": 0.04,
        "dst_host_srv_diff_host_rate": 1.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 1.0,
        "dst_host_srv_rerror_rate": 1.0,
    },
}


@app.route('/api/examples')
def examples():
    return jsonify(EXAMPLE_RECORDS)


if __name__ == '__main__':
    os.makedirs(IMG_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=5050, debug=False)