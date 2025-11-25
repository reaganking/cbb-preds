from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path("data/interstat/history/games_all.parquet")
OUT_DIR = Path("data/interstat/history")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "training_games.parquet"
OUT_CSV = OUT_DIR / "training_games_sample.csv"

REQUIRED_BASE = ["game_id", "date", "is_home", "pts", "opp_pts", "margin", "start_time"]

def _ensure_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee team_id / team_code / team_name exist; create fallbacks if missing."""
    df = df.copy()
    if "team_id" not in df.columns:
        if "team_code" in df.columns and df["team_code"].notna().any():
            df["team_id"] = df["team_code"].fillna("UNK")
        elif "team_name" in df.columns and df["team_name"].notna().any():
            df["team_id"] = df["team_name"].fillna("UNK")
        else:
            # last resort: build a synthetic id from game+home/away flag + row index
            df["team_id"] = (df["game_id"].astype(str) + "_" +
                             np.where(df.get("is_home", False), "H", "A"))
    if "team_code" not in df.columns:
        if "team_name" in df.columns:
            df["team_code"] = (
                df["team_name"].astype(str)
                .str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                .str[:6]
            )
        else:
            df["team_code"] = df["team_id"].astype(str)
    if "team_name" not in df.columns:
        df["team_name"] = df["team_code"].astype(str)
    return df

def _to_sort_key(df: pd.DataFrame) -> pd.Series:
    st = df.get("start_time", "00:00").fillna("00:00").astype(str)
    d = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.to_datetime(d + " " + st, errors="coerce")

def _add_team_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["date", "sort_key"], kind="mergesort").copy()

    # prior-game series
    s_pts = pd.to_numeric(df["pts"], errors="coerce").shift(1)
    s_opp = pd.to_numeric(df["opp_pts"], errors="coerce").shift(1)
    s_mgn = pd.to_numeric(df["margin"], errors="coerce").shift(1)

    df["gp_prev"] = np.arange(len(df))
    df[f"pts_mean_{window}"] = s_pts.rolling(window, min_periods=1).mean()
    df[f"opp_pts_mean_{window}"] = s_opp.rolling(window, min_periods=1).mean()
    df[f"margin_mean_{window}"] = s_mgn.rolling(window, min_periods=1).mean()

    df[f"pts_std_{window}"]  = s_pts.rolling(window, min_periods=1).std()
    df[f"opp_pts_std_{window}"] = s_opp.rolling(window, min_periods=1).std()
    df[f"margin_std_{window}"] = s_mgn.rolling(window, min_periods=1).std()

    # days rest (cap to [0,14], default 7 for first)
    rest = (df["date"] - df["date"].shift(1)).dt.days
    df["rest_days"] = rest.fillna(7).clip(lower=0, upper=14)
    return df

def main():
    df = pd.read_parquet(IN_PATH)

    # Minimal sanity + types
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    if missing:
        raise SystemExit(f"games_all.parquet missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ("pts", "opp_pts", "margin"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize booleans for is_home
    if df["is_home"].dtype != bool:
        # handles 1/0, "True"/"False", etc.
        df["is_home"] = df["is_home"].astype(str).str.lower().isin(["1","true","t","yes","y"])

    # Ensure team id/code/name are present
    df = _ensure_team_columns(df)

    # Sort key for stable within-day order
    df["sort_key"] = _to_sort_key(df)

    # Keep the columns we need for features (plus sort_key)
    keep = [
        "date", "game_id", "start_time", "is_home",
        "team_id", "team_code", "team_name",
        "pts", "opp_pts", "margin", "sort_key",
    ]
    df = df[keep].dropna(subset=["game_id", "date"]).copy()

    # Rolling features per team
    df = (
        df.groupby("team_id", group_keys=False)
          .apply(_add_team_rolling, window=5)
    )

    # Build home/away sides with distinct names
    base_cols = [
        "game_id", "date", "team_id", "team_code", "team_name",
        "pts", "opp_pts", "margin", "gp_prev", "rest_days",
        "pts_mean_5", "opp_pts_mean_5", "margin_mean_5",
        "pts_std_5", "opp_pts_std_5", "margin_std_5",
    ]
    h = df[df["is_home"]].loc[:, base_cols].copy()
    a = df[~df["is_home"]].loc[:, base_cols].copy()

    h.columns = [
        "game_id","date","home_team_id","home_team_code","home_team_name",
        "home_pts","home_opp_pts","home_margin","home_gp_prev","home_rest_days",
        "home_pts_mean_5","home_opp_pts_mean_5","home_margin_mean_5",
        "home_pts_std_5","home_opp_pts_std_5","home_margin_std_5",
    ]
    a.columns = [
        "game_id","date","away_team_id","away_team_code","away_team_name",
        "away_pts","away_opp_pts","away_margin","away_gp_prev","away_rest_days",
        "away_pts_mean_5","away_opp_pts_mean_5","away_margin_mean_5",
        "away_pts_std_5","away_opp_pts_std_5","away_margin_std_5",
    ]

    # Merge to single row per game
    g = pd.merge(h, a, on=["game_id", "date"], how="inner").copy()

    # Targets
    g["target_home_margin"] = g["home_pts"] - g["away_pts"]
    g["target_home_win"] = (g["target_home_margin"] > 0).astype(int)

    # Training eligibility (both teams have at least 3 prior games)
    eligible = (g["home_gp_prev"] >= 3) & (g["away_gp_prev"] >= 3)
    g_train = g.loc[eligible].sort_values(["date", "game_id"]).reset_index(drop=True)

    # Save
    g_train.to_parquet(OUT_PATH, index=False)
    g_train.head(50).to_csv(OUT_CSV, index=False)

    print(
        f"Saved training set: {OUT_PATH}  rows={len(g_train)}  from_games={g_train['game_id'].nunique()}"
    )
    print(f"Sample CSV: {OUT_CSV}")

if __name__ == "__main__":
    main()