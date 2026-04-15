"""
packet_capture.py
Live packet capture using Scapy → feature extraction → IDS prediction.

Usage (must run as root/sudo):
    sudo python packet_capture.py --iface eth0 --count 100
    sudo python packet_capture.py --iface eth0 --live   # continuous

Requires:
    pip install scapy
    # Linux: may also need  apt install libpcap-dev
    # macOS: works out of the box with sudo
"""

import argparse, os, sys, time, json
from datetime import datetime
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(__file__))

# ── Optional imports — degrade gracefully ────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("[capture] WARNING: Scapy not installed — live capture disabled.")
    print("          pip install scapy")

try:
    import joblib
    import numpy as np
    from scripts.preprocess import load_artifacts, encode_and_scale
    import pandas as pd
    MODELS_OK = True
except Exception as e:
    MODELS_OK = False
    print(f"[capture] WARNING: Model loading unavailable — {e}")

MDL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction: Scapy packet → NSL-KDD-compatible feature dict
# ─────────────────────────────────────────────────────────────────────────────

# Per-connection state tracker  {(src,dst,dport): state_dict}
_flow_state = defaultdict(lambda: {
    'start': None, 'src_bytes': 0, 'dst_bytes': 0,
    'count': 0, 'serror_count': 0, 'flag_hist': [],
})

# Rolling window for count / rate features
_recent_connections = deque(maxlen=100)   # last 100 connection records


def _get_service(dport):
    PORT_MAP = {
        80: 'http', 443: 'http', 21: 'ftp', 20: 'ftp_data',
        25: 'smtp', 22: 'ssh', 53: 'dns', 110: 'pop3',
        119: 'nntp', 23: 'telnet', 79: 'finger', 113: 'auth',
    }
    return PORT_MAP.get(dport, 'other')


def _get_flag(pkt):
    """Map TCP flags to NSL-KDD flag string."""
    if not pkt.haslayer(TCP):
        return 'OTH'
    flags = pkt[TCP].flags
    s = str(flags)
    if 'S' in s and 'A' in s and 'F' in s:
        return 'SF'
    elif 'S' in s and 'A' not in s:
        return 'S0'
    elif 'R' in s and 'S' in s and 'T' in s:
        return 'RSTO'
    elif 'R' in s:
        return 'REJ'
    elif 'S' in s and 'H' in s:
        return 'SH'
    return 'OTH'


def extract_features(pkt) -> dict | None:
    """
    Extract NSL-KDD-compatible features from a single Scapy packet.
    Returns None if the packet isn't IP.
    """
    if not pkt.haslayer(IP):
        return None

    ip  = pkt[IP]
    now = time.time()

    # Protocol
    if pkt.haslayer(TCP):
        proto = 'tcp'
        dport = pkt[TCP].dport
        sport = pkt[TCP].sport
    elif pkt.haslayer(UDP):
        proto = 'udp'
        dport = pkt[UDP].dport
        sport = pkt[UDP].sport
    elif pkt.haslayer(ICMP):
        proto = 'icmp'
        dport = 0
        sport = 0
    else:
        return None

    src, dst = ip.src, ip.dst
    flow_key = (src, dst, dport)
    state    = _flow_state[flow_key]

    # Init flow
    if state['start'] is None:
        state['start'] = now

    duration = max(0, now - state['start'])

    # Byte counts (approximate per-packet)
    pkt_len = len(pkt)
    src_bytes = state['src_bytes'] + pkt_len
    dst_bytes = state['dst_bytes']  # simplified
    state['src_bytes'] = src_bytes
    state['count']    += 1

    # Rolling window features
    _recent_connections.append({
        'dst': dst, 'dport': dport, 'proto': proto, 'time': now,
    })
    window    = list(_recent_connections)
    cnt       = sum(1 for c in window if c['dst'] == dst)
    srv_cnt   = sum(1 for c in window if c['dport'] == dport)
    total     = max(len(window), 1)
    same_srv  = srv_cnt / total
    diff_srv  = 1 - same_srv
    serr_rate = 0.0   # simplified — would need SYN-no-reply tracking

    service = _get_service(dport)
    flag    = _get_flag(pkt)

    return {
        'duration':                  int(duration),
        'protocol_type':             proto,
        'service':                   service,
        'flag':                      flag,
        'src_bytes':                 src_bytes,
        'dst_bytes':                 dst_bytes,
        'land':                      int(src == dst),
        'wrong_fragment':            0,
        'urgent':                    0,
        'hot':                       0,
        'num_failed_logins':         0,
        'logged_in':                 0,
        'num_compromised':           0,
        'root_shell':                0,
        'su_attempted':              0,
        'num_root':                  0,
        'num_file_creations':        0,
        'num_shells':                0,
        'num_access_files':          0,
        'num_outbound_cmds':         0,
        'is_host_login':             0,
        'is_guest_login':            0,
        'count':                     cnt,
        'srv_count':                 srv_cnt,
        'serror_rate':               serr_rate,
        'srv_serror_rate':           serr_rate,
        'rerror_rate':               0.0,
        'srv_rerror_rate':           0.0,
        'same_srv_rate':             round(same_srv, 2),
        'diff_srv_rate':             round(diff_srv, 2),
        'srv_diff_host_rate':        0.0,
        'dst_host_count':            min(dst_bytes // 100 + 1, 255),
        'dst_host_srv_count':        srv_cnt,
        'dst_host_same_srv_rate':    round(same_srv, 2),
        'dst_host_diff_srv_rate':    round(diff_srv, 2),
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate':      serr_rate,
        'dst_host_srv_serror_rate':  serr_rate,
        'dst_host_rerror_rate':      0.0,
        'dst_host_srv_rerror_rate':  0.0,
        # metadata (not fed to model)
        '_src': src, '_dst': dst, '_dport': dport, '_time': now,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────────────

class IDSPredictor:
    def __init__(self, model_dir=MDL_DIR):
        if not MODELS_OK:
            raise RuntimeError("Model dependencies not available")
        self.scaler, self.feature_cols, self.le = load_artifacts(model_dir)
        self.rf = joblib.load(os.path.join(model_dir, 'rf_binary.pkl'))
        print(f"[capture] Loaded RF model + scaler from {model_dir}/")

    def predict(self, feature_dict: dict) -> dict:
        meta_keys = {k for k in feature_dict if k.startswith('_')}
        clean     = {k: v for k, v in feature_dict.items() if k not in meta_keys}
        clean['label'] = 'normal'  # placeholder — preprocess needs it

        df = pd.DataFrame([clean])
        X, _, _ = encode_and_scale(df, scaler=self.scaler,
                                    feature_cols=self.feature_cols, fit=False)
        pred  = self.rf.predict(X)[0]
        proba = self.rf.predict_proba(X)[0][1]
        return {
            'prediction':  'ATTACK' if pred else 'NORMAL',
            'confidence':  round(float(proba), 4),
            'src':         feature_dict.get('_src'),
            'dst':         feature_dict.get('_dst'),
            'dport':       feature_dict.get('_dport'),
            'timestamp':   datetime.fromtimestamp(
                               feature_dict.get('_time', time.time())
                           ).isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sniff loop
# ─────────────────────────────────────────────────────────────────────────────

def capture_and_classify(iface='eth0', count=0, output_log='data/capture_log.jsonl'):
    """
    Capture live packets, extract features, run IDS model, log results.
    count=0 → run forever (Ctrl+C to stop).
    """
    if not SCAPY_OK:
        print("Scapy unavailable — cannot capture live packets.")
        return

    predictor = IDSPredictor()
    os.makedirs(os.path.dirname(output_log) or '.', exist_ok=True)
    log_file  = open(output_log, 'a')
    alerts    = 0
    processed = 0

    print(f"\n[capture] Sniffing on {iface}  (count={'∞' if count==0 else count})")
    print("[capture] Press Ctrl+C to stop\n")

    def handle(pkt):
        nonlocal alerts, processed
        feat = extract_features(pkt)
        if feat is None:
            return
        result = predictor.predict(feat)
        processed += 1
        log_file.write(json.dumps(result) + '\n')
        log_file.flush()

        if result['prediction'] == 'ATTACK':
            alerts += 1
            ts  = result['timestamp'][11:19]
            src = result['src']
            dst = result['dst']
            dpt = result['dport']
            conf= result['confidence']
            print(f"  🚨  [{ts}] ATTACK detected  {src} → {dst}:{dpt}  "
                  f"(conf={conf:.2%})  [total alerts: {alerts}]")
        else:
            if processed % 50 == 0:
                print(f"  ✓   [{processed:>6} pkts]  {alerts} alerts so far")

    try:
        sniff(iface=iface, prn=handle, count=count, store=False)
    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()

    print(f"\n[capture] Done.  Processed {processed} packets, {alerts} alerts.")
    print(f"[capture] Log → {output_log}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDS Live Packet Capture')
    parser.add_argument('--iface',  default='eth0',
                        help='Network interface (default: eth0)')
    parser.add_argument('--count',  type=int, default=0,
                        help='Packets to capture (0=infinite)')
    parser.add_argument('--log',    default='../data/capture_log.jsonl',
                        help='Output JSONL log path')
    args = parser.parse_args()

    capture_and_classify(iface=args.iface, count=args.count, output_log=args.log)