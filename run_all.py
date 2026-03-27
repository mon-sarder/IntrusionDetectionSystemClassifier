#!/usr/bin/env python3
"""
run_all.py — One-shot runner for the full IDS pipeline.

Steps:
  1. Generate / load dataset
  2. Train classical ML models (RF + SVM) + generate all plots
  3. (optional) Train LSTM
  4. Launch Flask dashboard

Usage:
  python run_all.py                  # full pipeline, skip LSTM
  python run_all.py --with-lstm      # include LSTM training
  python run_all.py --skip-train     # just launch the dashboard
  python run_all.py --port 8080      # custom port
"""
import argparse, os, sys, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, 'scripts')
sys.path.insert(0, SCRIPTS)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       Intrusion Detection System — Full Pipeline         ║
║       NSL-KDD  |  RF + SVM + LSTM  |  Flask Dashboard   ║
╚══════════════════════════════════════════════════════════╝
"""


def run_step(label, fn):
    print(f"\n{'─'*60}")
    print(f"  ▶  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    print(f"\n  ✓  {label} done  ({elapsed:.1f}s)")


def step_data():
    from generate_data import load_dataset
    df, _, src = load_dataset(os.path.join(ROOT, 'data'))
    print(f"Dataset: {src}  |  {len(df):,} rows")


def step_train_classical():
    from train_classical import main
    main()


def step_train_lstm(seq_len=10, epochs=20):
    try:
        from train_lstm import main
        main(seq_len=seq_len, epochs=epochs, mode='binary')
    except SystemExit:
        print("  [!] LSTM skipped — TensorFlow not installed")
        print("      pip install tensorflow   to enable")


def step_launch_flask(port=5050):
    import flask
    print(f"\n{'─'*60}")
    print(f"  ▶  Launching IDS Dashboard on http://localhost:{port}")
    print(f"{'─'*60}")
    os.chdir(ROOT)
    # Import and run inline (single-process, good for dev)
    app_path = os.path.join(ROOT, 'app.py')
    os.environ['FLASK_ENV'] = 'development'
    import importlib.util
    spec = importlib.util.spec_from_file_location('app', app_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-lstm',   action='store_true', help='Also train LSTM')
    parser.add_argument('--skip-train',  action='store_true', help='Skip training, just launch dashboard')
    parser.add_argument('--port',        type=int, default=5050)
    parser.add_argument('--lstm-epochs', type=int, default=20)
    parser.add_argument('--lstm-seqlen', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, 'data'),       exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'models'),     exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'static/img'), exist_ok=True)

    if not args.skip_train:
        run_step("1/3  Load / generate dataset",         step_data)
        run_step("2/3  Train RF + SVM + generate plots", step_train_classical)
        if args.with_lstm:
            run_step("3/3  Train LSTM",
                     lambda: step_train_lstm(args.lstm_seqlen, args.lstm_epochs))
        else:
            print("\n  ℹ  Skipping LSTM — pass --with-lstm to enable")
    else:
        print("  ℹ  Skipping training (--skip-train)")

    step_launch_flask(args.port)


if __name__ == '__main__':
    main()