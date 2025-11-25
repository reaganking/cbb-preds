# src/pipeline/ingest_range.py
import argparse, sys, subprocess, os
from datetime import datetime, timedelta
import pandas as pd

def daterange(start_date: str, end_date: str):
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    d = s
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)

def run_daily(module: str, day: str, daily_dir: str):
    """Runs your existing daily fetcher module which should write a parquet for that day."""
    # Expectation: module accepts a single arg (YYYY-MM-DD) and writes to daily_dir/{day}.parquet
    env = os.environ.copy()
    # Ensure local src is importable
    if "." not in sys.path:
        env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
        if env["PYTHONPATH"] == ".":
            pass
        else:
            env["PYTHONPATH"] = f".:{env['PYTHONPATH']}"
    print(f"[ingest] fetching {day} via {module} ...")
    subprocess.run([sys.executable, "-m", module, day, "--out-dir", daily_dir], check=True)

def build_historical(daily_dir: str, historical_path: str):
    print(f"[ingest] building historical from {daily_dir} -> {historical_path}")
    files = sorted([
        os.path.join(daily_dir, f)
        for f in os.listdir(daily_dir)
        if f.endswith(".parquet")
    ])
    if not files:
        print("[ingest] no daily files found")
        return

    # Read in chunks to keep memory light
    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            dfs.append(df)
        except Exception as e:
            print(f"[ingest] skip {fp} ({e})")

    if not dfs:
        print("[ingest] nothing readable")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Canonical keys + light cleanup
    required = ["date", "game_id", "team_id"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"expected column '{col}' in daily parquet")

    # De-dupe in case you re-fetch a day
    df = df.drop_duplicates(subset=["date", "game_id", "team_id"]).sort_values(["date","game_id","team_id"])

    # Strong dtypes where reasonable
    for c in ["pts","opp_pts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Write historical
    os.makedirs(os.path.dirname(historical_path), exist_ok=True)
    df.to_parquet(historical_path, index=False)
    print(f"[ingest] wrote {len(df):,} rows to {historical_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--module", default="src.data.fetch_interstat",
                    help="module to run for a single day fetch")
    ap.add_argument("--daily-dir", default="data/interstat/daily",
                    help="where daily parquet files are written")
    ap.add_argument("--historical-path", default="data/interstat/historical.parquet",
                    help="output historical parquet path")
    ap.add_argument("--refresh-daily", action="store_true",
                    help="re-pull even if a daily file exists")
    args = ap.parse_args()

    os.makedirs(args.daily_dir, exist_ok=True)

    # 1) fetch all days
    for day in daterange(args.start, args.end):
        out_fp = os.path.join(args.daily_dir, f"{day}.parquet")
        if os.path.exists(out_fp) and not args.refresh_daily:
            print(f"[ingest] {day} exists -> {out_fp} (skip)")
            continue
        run_daily(args.module, day, args.daily_dir)

    # 2) build historical
    build_historical(args.daily_dir, args.historical_path)

if __name__ == "__main__":
    main()