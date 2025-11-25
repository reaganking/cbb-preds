import os, glob, pandas as pd
from datetime import datetime
from pathlib import Path

def season_start_year(dt: pd.Timestamp) -> int:
    return dt.year if dt.month >= 11 else dt.year - 1

def main():
    daily_dir = "data/interstat/daily"
    out_dir = "data/interstat/history"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(daily_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {daily_dir}")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # normalize dtypes
    df["date"] = pd.to_datetime(df["date"])
    df["is_home"] = df["is_home"].astype(bool)
    df["team_id"] = df["team_id"].astype(str)
    df["opp_id"] = df["opp_id"].astype(str)

    # drop dupes (if any)
    df = df.drop_duplicates(subset=["date","game_id","team_id","is_home"]).sort_values(["date","game_id","is_home"]).reset_index(drop=True)

    # add season key like 2024 for 2024-25 season
    df["season_start"] = df["date"].apply(season_start_year)

    out_path = os.path.join(out_dir, "games_all.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved games table: {out_path}  rows={len(df)}  games={(len(df)//2)}")

if __name__ == "__main__":
    main()