# src/data/build_history.py
import argparse, os, glob, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

KEEP_COLS = [
    "date","game_id","status","start_time","overtime",
    "venue_id","venue_name","citystate","neutral","attendance",
    "siteurl","apiurl",
    "is_home","team_id","team_code","team_name",
    "opp_id","opp_code","opp_name",
    "pts","opp_pts","margin","pbp_count","playerstatlines_count",
]

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected columns exist
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[KEEP_COLS].copy()

    # Basic type cleanups
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in ["game_id","venue_id","team_id","opp_id","pbp_count","playerstatlines_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["pts","opp_pts","margin","attendance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Booleans sometimes arrive as strings
    df["neutral"] = df["neutral"].astype("boolean")
    df["is_home"] = df["is_home"].astype("boolean")

    # Strings
    for c in ["status","start_time","overtime","venue_name","citystate",
              "siteurl","apiurl","team_code","team_name","opp_code","opp_name"]:
        df[c] = df[c].astype("string")

    return df

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-dir", default="data/interstat/daily")
    ap.add_argument("--out-file", default="data/interstat/history/games_teams.parquet")
    args = ap.parse_args()

    daily_dir = Path(args.daily_dir)
    files = sorted([*glob.glob(str(daily_dir / "*.parquet")),
                    *glob.glob(str(daily_dir / "*.csv"))])

    if not files:
        print(f"No daily files found in {daily_dir}")
        return

    frames = []
    for f in files:
        try:
            df = read_any(f)
            df = coerce_types(df)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    hist = pd.concat(frames, ignore_index=True)

    # Deduplicate on team-game identity
    hist = hist.drop_duplicates(subset=["date","game_id","team_id","is_home"], keep="first")
    hist = hist.sort_values(["date","game_id","is_home","team_id"]).reset_index(drop=True)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    # Write via pyarrow to keep types clean
    table = pa.Table.from_pandas(hist)
    pq.write_table(table, args.out_file)

    n_games = hist[["date","game_id"]].drop_duplicates().shape[0]
    print(f"Saved {len(hist):,} team-rows across {n_games:,} games -> {args.out_file}")

if __name__ == "__main__":
    main()