"""
generate_data.py
Generates a large NSL-KDD-style synthetic dataset (or loads the real one).
Drop KDDTrain+.txt / KDDTest+.txt into data/ to use the real dataset.
"""
import numpy as np
import pandas as pd
import os

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
]

PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES  = ['http','ftp','smtp','ssh','dns','pop3','nntp','telnet','ftp_data',
              'finger','auth','eco_i','ecr_i','other','private','urp_i','red_i']
FLAGS     = ['SF','S0','REJ','RSTO','RSTR','SH','S1','S2','S3','OTH']

PW_DEFAULT = [0.60, 0.30, 0.10]
FW_DEFAULT = [0.810,0.050,0.040,0.030,0.020,0.015,0.013,0.010,0.007,0.005]
FW_DOS     = [0.100,0.700,0.099,0.040,0.020,0.015,0.010,0.008,0.005,0.003]


def _make(n, label, attack_cat, dur, src, dst, serr, rerr, ssrv, cnt,
          fw=None, pw=None, overrides=None):
    rng = np.random
    fw = fw or FW_DEFAULT
    pw = pw or PW_DEFAULT
    d = {
        'duration':              rng.randint(*dur, n),
        'protocol_type':         rng.choice(PROTOCOLS, n, p=pw),
        'service':               rng.choice(SERVICES, n),
        'flag':                  rng.choice(FLAGS, n, p=fw),
        'src_bytes':             rng.randint(*src, n),
        'dst_bytes':             rng.randint(*dst, n),
        'land':                  (rng.rand(n) < 0.001).astype(int),
        'wrong_fragment':        (rng.rand(n) < 0.005).astype(int),
        'urgent':                np.zeros(n, int),
        'hot':                   rng.randint(0, 10, n),
        'num_failed_logins':     rng.randint(0, 2, n),
        'logged_in':             (rng.rand(n) < 0.60).astype(int),
        'num_compromised':       rng.randint(0, 3, n),
        'root_shell':            np.zeros(n, int),
        'su_attempted':          np.zeros(n, int),
        'num_root':              np.zeros(n, int),
        'num_file_creations':    np.zeros(n, int),
        'num_shells':            np.zeros(n, int),
        'num_access_files':      np.zeros(n, int),
        'num_outbound_cmds':     np.zeros(n, int),
        'is_host_login':         np.zeros(n, int),
        'is_guest_login':        (rng.rand(n) < 0.01).astype(int),
        'count':                 rng.randint(*cnt, n),
        'srv_count':             rng.randint(1, 200, n),
        'serror_rate':           rng.uniform(*serr, n).round(2),
        'srv_serror_rate':       rng.uniform(*serr, n).round(2),
        'rerror_rate':           rng.uniform(*rerr, n).round(2),
        'srv_rerror_rate':       rng.uniform(*rerr, n).round(2),
        'same_srv_rate':         rng.uniform(*ssrv, n).round(2),
        'diff_srv_rate':         rng.uniform(0, 0.20, n).round(2),
        'srv_diff_host_rate':    rng.uniform(0, 0.30, n).round(2),
        'dst_host_count':        rng.randint(1, 255, n),
        'dst_host_srv_count':    rng.randint(1, 255, n),
        'dst_host_same_srv_rate':       rng.uniform(0.1, 1.0, n).round(2),
        'dst_host_diff_srv_rate':       rng.uniform(0.0, 0.2, n).round(2),
        'dst_host_same_src_port_rate':  rng.uniform(0.0, 1.0, n).round(2),
        'dst_host_srv_diff_host_rate':  rng.uniform(0.0, 0.3, n).round(2),
        'dst_host_serror_rate':         rng.uniform(*serr, n).round(2),
        'dst_host_srv_serror_rate':     rng.uniform(*serr, n).round(2),
        'dst_host_rerror_rate':         rng.uniform(*rerr, n).round(2),
        'dst_host_srv_rerror_rate':     rng.uniform(*rerr, n).round(2),
        'label':                 label,
        'attack_category':       attack_cat,
        'difficulty_level':      rng.randint(1, 21, n),
    }
    df = pd.DataFrame(d)
    if overrides:
        for col, vals in overrides.items():
            df[col] = vals
    return df


def generate_synthetic(n_total=20000, seed=42):
    np.random.seed(seed)
    # Class proportions mirroring NSL-KDD
    n_normal = int(n_total * 0.535)
    n_dos    = int(n_total * 0.275)
    n_probe  = int(n_total * 0.110)
    n_r2l    = int(n_total * 0.054)
    n_u2r    = n_total - n_normal - n_dos - n_probe - n_r2l

    rng = np.random

    normal = _make(n_normal,'normal','Normal',(0,200),(100,60000),(100,60000),
                   (0.00,0.05),(0.00,0.05),(0.70,1.00),(1,100))

    dos = _make(n_dos,'dos','DoS',(0,5),(0,1500),(0,200),
                (0.70,1.00),(0.00,0.10),(0.90,1.00),(200,511),fw=FW_DOS)
    dos['count'] = rng.randint(400, 511, n_dos)

    probe = _make(n_probe,'probe','Probe',(0,30),(0,800),(0,800),
                  (0.10,0.50),(0.30,0.80),(0.10,0.40),(1,60))
    probe['diff_srv_rate'] = rng.uniform(0.5, 1.0, n_probe).round(2)
    probe['dst_host_diff_srv_rate'] = rng.uniform(0.4, 1.0, n_probe).round(2)

    r2l = _make(n_r2l,'r2l','R2L',(50,6000),(100,8000),(1000,150000),
                (0.00,0.10),(0.00,0.10),(0.40,0.90),(1,40))
    r2l['num_failed_logins'] = rng.randint(1, 6, n_r2l)
    r2l['logged_in']         = rng.choice([0,1], n_r2l, p=[0.3,0.7])

    u2r = _make(n_u2r,'u2r','U2R',(100,12000),(1000,25000),(1000,25000),
                (0.00,0.05),(0.00,0.05),(0.30,0.80),(1,15))
    u2r['root_shell']    = rng.choice([0,1], n_u2r, p=[0.2,0.8])
    u2r['su_attempted']  = rng.choice([0,1], n_u2r, p=[0.4,0.6])
    u2r['num_root']      = rng.randint(0, 6, n_u2r)
    u2r['num_shells']    = rng.randint(0, 3, n_u2r)

    df = pd.concat([normal, dos, probe, r2l, u2r], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def load_real_nslkdd(train_path, test_path=None):
    """Load the real NSL-KDD .txt files (no header)."""
    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test  = pd.read_csv(test_path,  header=None, names=COLUMNS) if test_path else None
    return df_train, df_test


def load_dataset(data_dir='data'):
    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path  = os.path.join(data_dir, 'KDDTest+.txt')

    if os.path.exists(train_path):
        print(f"[data] Real NSL-KDD found at {train_path}")
        train, test = load_real_nslkdd(train_path, test_path if os.path.exists(test_path) else None)
        return train, test, 'real'
    else:
        print("[data] Real NSL-KDD not found — using synthetic dataset")
        print("       Drop KDDTrain+.txt (and KDDTest+.txt) into data/ to use real data")
        df = generate_synthetic(n_total=20000)
        out = os.path.join(data_dir, 'nsl_kdd_synthetic.csv')
        df.to_csv(out, index=False)
        print(f"[data] Synthetic dataset saved → {out}")
        return df, None, 'synthetic'


if __name__ == '__main__':
    train, test, src = load_dataset()
    print(f"Source: {src}")
    print(f"Train shape: {train.shape}")
    print(train['label'].value_counts())