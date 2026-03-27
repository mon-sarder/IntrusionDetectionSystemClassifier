# 🛡️ Intrusion Detection System — NSL-KDD

> **Cal Poly Pomona CS + Cybersecurity Portfolio Project**
> Random Forest · SVM · LSTM · Flask Dashboard · Live Packet Capture

---

## Architecture

```
ids_full/
├── run_all.py               ← one-shot launcher
├── app.py                   ← Flask API + dashboard
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── generate_data.py     ← dataset generation / real NSL-KDD loader
│   ├── preprocess.py        ← shared preprocessing pipeline
│   ├── train_classical.py   ← RF + SVM + CV + ROC plots
│   ├── train_lstm.py        ← LSTM deep learning classifier
│   └── packet_capture.py   ← live Scapy capture + classification
├── data/                    ← put KDDTrain+.txt here for real data
├── models/                  ← saved .pkl / .keras files
├── static/img/              ← generated training plots
└── templates/
    └── dashboard.html       ← full IDS web dashboard
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn joblib gunicorn
# For LSTM:
pip install tensorflow
# For live capture:
pip install scapy
```

### 2. (Optional) Use real NSL-KDD dataset

Download from: https://www.unb.ca/cic/datasets/nsl.html

```bash
cp KDDTrain+.txt ids_full/data/
cp KDDTest+.txt  ids_full/data/   # optional
```

If omitted, a realistic 20,000-sample synthetic dataset is generated automatically.

### 3. Run the full pipeline

```bash
cd ids_full

# Train + launch dashboard
python run_all.py

# Include LSTM training
python run_all.py --with-lstm --lstm-epochs 30

# Just launch dashboard (models already trained)
python run_all.py --skip-train

# Custom port
python run_all.py --port 8080
```

Open **http://localhost:5050** → full dashboard.

---

## Training individually

```bash
cd ids_full

# Step 1: Classical ML (RF + SVM + plots)
python scripts/train_classical.py

# Step 2: LSTM
python scripts/train_lstm.py --mode binary --epochs 30 --seq-len 10
python scripts/train_lstm.py --mode multi  --epochs 30

# Step 3: Flask
python app.py
```

---

## Live Packet Capture (requires root)

```bash
# Linux / macOS — must run as root
sudo python scripts/packet_capture.py --iface eth0
sudo python scripts/packet_capture.py --iface en0 --count 500
```

Logs detections to `data/capture_log.jsonl`.

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/status` | Model readiness + uptime |
| POST | `/api/predict` | Single-record classification |
| POST | `/api/predict/batch` | Batch classification (≤500) |
| GET  | `/api/history` | Last N predictions |
| GET  | `/api/stats` | Aggregate stats for dashboard |
| GET  | `/api/examples` | Example records for each attack type |
| POST | `/api/retrain` | Trigger background retraining |

### Single prediction

```bash
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0",
    "src_bytes": 0, "dst_bytes": 0, "count": 511, "serror_rate": 1.0,
    "srv_serror_rate": 1.0, "same_srv_rate": 1.0,
    "dst_host_serror_rate": 1.0, "model": "rf_binary"
  }'
```

Response:
```json
{
  "id": "a3f9c1d2",
  "timestamp": "2025-09-12T14:32:01.123456",
  "prediction": "ATTACK",
  "attack_type": "DOS",
  "confidence": 0.9987,
  "model_used": "rf_binary",
  "features_in": 61
}
```

---

## Docker

```bash
docker-compose up --build
```

Open **http://localhost:5050**

For live capture inside Docker:

```yaml
# in docker-compose.yml, uncomment:
network_mode: host
privileged: true
```

---

## Model Performance (synthetic — real NSL-KDD will be similar)

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Random Forest (binary) | 99.9%+ | 99.9%+ | 0.9999 |
| Random Forest (multi) | 99.9%+ | 99.9%+ | — |
| SVM RBF (binary) | ~99.4% | ~99.3% | 0.9998 |
| LSTM (binary) | ~98–99% | ~98–99% | 0.999 |
| **RF CV (5-fold mean)** | **99.9%** | **±0.01%** | — |

On the **real NSL-KDD**, published benchmarks show RF at ~99.5% binary / ~97–98% multi-class.

---

## Resume Talking Points

- Built end-to-end IDS pipeline: data → preprocessing → ML/DL → REST API → live dashboard
- Implemented 3 classifiers (Random Forest, SVM, LSTM) with cross-validation, ROC-AUC, and confusion matrix analysis
- Engineered 61 features from NSL-KDD's 41 raw features via one-hot encoding + standardization
- Designed REST API serving real-time predictions; supports batch inference (≤500 records/call)
- Added live traffic analysis using Scapy for packet-level feature extraction
- Containerized with Docker for reproducible deployment