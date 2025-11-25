from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

HIST_GAMES = Path("data/interstat/history/games_all.parquet")
DAILY_DIR = Path("data/interstat/daily")
OUT_DIR = Path("data/interstat/history")
MODELS_DIR = Path("models")


# ---------- helpers ----------
def _add_team_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling team features within a single team's history."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "game_id"], kind="mergesort")

    # rolling means/stds of points and margin using prior games only (shift)
    for col in ("pts", "opp_pts"):
        roll = df[col].rolling(window, min_periods=1)
        df[f"{col}_mean_{window}"] = roll.mean().shift(1)
        df[f"{col}_std_{window}"] = roll.std().shift(1)
    df[f"margin_mean_{window}"] = (df["pts"] - df["opp_pts"]).rolling(window, min_periods=1).mean().shift(1)

    # rest days since prior game
    df["rest_days"] = (df["date"] - df["date"].shift(1)).dt.days
    df["rest_days"] = df["rest_days"].fillna(7).clip(lower=0)

    # prior games played (0 for first game, 1 for second, etc.)
    df["gp_prev"] = pd.Series(range(len(df)), index=df.index).astype(float)

    return df


def _prefix_except(df: pd.DataFrame, prefix: str, skip: tuple[str, ...] = ("team_id",)) -> pd.DataFrame:
    """Add a prefix to all columns except those in `skip`."""
    rename_map = {c: f"{prefix}{c}" for c in df.columns if c not in skip}
    return df.rename(columns=rename_map)


def prob_to_american(p: float) -> float:
    if p is None or np.isnan(p) or p <= 0 or p >= 1:
        return np.nan
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def build_pairs_for_date(target_date: str) -> pd.DataFrame:
    """
    Build one row per game for `target_date` with home/away IDs and labels,
    reading daily file from parquet (preferred) or csv fallback.
    """
    import os

    base = DAILY_DIR / target_date
    f_parq = f"{base}.parquet"
    f_csv = f"{base}.csv"

    if os.path.exists(f_parq):
        df = pd.read_parquet(f_parq)
    elif os.path.exists(f_csv):
        df = pd.read_csv(f_csv)
    else:
        raise FileNotFoundError(f"Missing daily file for {target_date}: {f_parq} or {f_csv}")

    # Expect one row per team per game with `is_home`
    home = df[df["is_home"]].rename(
        columns={"team_id": "home_team_id", "team_code": "home_team_code", "team_name": "home_team_name"}
    )
    away = df[~df["is_home"]].rename(
        columns={"team_id": "away_team_id", "team_code": "away_team_code", "team_name": "away_team_name"}
    )

    # Start from the home rows, use opp_* if present, then overwrite with away table if available
    pairs = home[
        [
            "date",
            "game_id",
            "home_team_id",
            "home_team_code",
            "home_team_name",
            "opp_id",
            "opp_code",
            "opp_name",
        ]
    ].rename(
        columns={
            "opp_id": "away_team_id",
            "opp_code": "away_team_code",
            "opp_name": "away_team_name",
        }
    )

    pairs = pairs.merge(
        away[["game_id", "away_team_id", "away_team_code", "away_team_name"]],
        on="game_id",
        how="left",
        suffixes=("", "_awaytbl"),
    )

    for col in ("away_team_id", "away_team_code", "away_team_name"):
        aux = f"{col}_awaytbl"
        if aux in pairs:
            pairs[col] = pairs[col].fillna(pairs[aux])
            pairs.drop(columns=[aux], inplace=True)

    # Normalize dtypes for future merges
    for col in ("home_team_id", "away_team_id"):
        if col in pairs:
            pairs[col] = pairs[col].astype("string")

    # Keep core columns only
    return pairs[
        [
            "date",
            "game_id",
            "home_team_id",
            "home_team_code",
            "home_team_name",
            "away_team_id",
            "away_team_code",
            "away_team_name",
        ]
    ].copy()


def _features_from_pairs(pairs: pd.DataFrame, asof_roll: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join latest per-team rolling features to home/away teams and build model matrix X.
    """
    # Ensure IDs are strings to avoid int/object merge errors
    pairs = pairs.copy()
    for col in ("home_team_id", "away_team_id"):
        pairs[col] = pairs[col].astype("string")

    asof = asof_roll.copy()
    asof["team_id"] = asof["team_id"].astype("string")

    home_feats = _prefix_except(asof, "home_", skip=("team_id",))
    away_feats = _prefix_except(asof, "away_", skip=("team_id",))

    g = pairs.merge(home_feats, left_on="home_team_id", right_on="team_id", how="left").drop(columns=["team_id"])
    g = g.merge(away_feats, left_on="away_team_id", right_on="team_id", how="left").drop(columns=["team_id"])

    # Coerce numeric feature columns and build differences/sums
    num_cols = [
        "pts_mean_5",
        "opp_pts_mean_5",
        "margin_mean_5",
        "pts_std_5",
        "opp_pts_std_5",
        "rest_days",
        "gp_prev",
    ]
    for c in num_cols:
        g[f"home_{c}"] = pd.to_numeric(g.get(f"home_{c}"), errors="coerce")
        g[f"away_{c}"] = pd.to_numeric(g.get(f"away_{c}"), errors="coerce")

    X = pd.DataFrame(index=g.index)
    X["diff_pts_mean_5"] = g["home_pts_mean_5"] - g["away_pts_mean_5"]
    X["diff_opp_pts_mean_5"] = g["home_opp_pts_mean_5"] - g["away_opp_pts_mean_5"]
    X["diff_margin_mean_5"] = g["home_margin_mean_5"] - g["away_margin_mean_5"]
    X["diff_pts_std_5"] = g["home_pts_std_5"] - g["away_pts_std_5"]
    X["diff_opp_pts_std_5"] = g["home_opp_pts_std_5"] - g["away_opp_pts_std_5"]
    X["diff_rest_days"] = g["home_rest_days"] - g["away_rest_days"]
    X["sum_gp_prev"] = g["home_gp_prev"] + g["away_gp_prev"]
    X = X.fillna(0.0)

    return X, g


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD to predict")
    ap.add_argument("--hist", default=str(HIST_GAMES))
    ap.add_argument("--models", default=str(MODELS_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    date_target = pd.Timestamp(args.date).normalize()

    # Load game history and build as-of (day-1) rolling features
    games = pd.read_parquet(args.hist)
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()

    hist = games.loc[games["date"] < date_target].copy()

    roll = (
        hist.sort_values(["team_id", "date", "game_id"])
        .groupby("team_id", group_keys=False)
        .apply(_add_team_rolling, window=5)
    )

    asof = (
        roll.sort_values(["team_id", "date", "game_id"])
        .groupby("team_id", as_index=False)
        .tail(1)
        .loc[
            :,
            [
                "team_id",
                "pts_mean_5",
                "opp_pts_mean_5",
                "margin_mean_5",
                "pts_std_5",
                "opp_pts_std_5",
                "rest_days",
                "gp_prev",
            ],
        ]
        .reset_index(drop=True)
    )

    # Build today’s game pairs and features
    pairs = build_pairs_for_date(args.date)
    X, g = _features_from_pairs(pairs, asof)

    # Load baselines and predict
    m_margin = load(Path(args.models) / "baseline_margin.joblib")
    m_win = load(Path(args.models) / "baseline_win.joblib")

    pred_home_margin = m_margin.predict(X).astype(float)
    prob_home = m_win.predict_proba(X)[:, 1].astype(float)

    # Assemble board
    out = g.loc[
        :,
        [
            "date",
            "game_id",
            "home_team_code",
            "home_team_name",
            "away_team_code",
            "away_team_name",
        ],
    ].copy()

    out["pred_home_margin"] = pred_home_margin
    out["home_spread"] = -pred_home_margin  # negative -> home favored
    out["prob_home_win"] = prob_home
    out["home_moneyline_nv"] = np.rint([prob_to_american(p) for p in prob_home]).astype("float")
    out["away_moneyline_nv"] = np.rint([prob_to_american(1.0 - p) for p in prob_home]).astype("float")

    out = out.sort_values(["date", "game_id"]).reset_index(drop=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"board_{args.date}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved board -> {out_path}  rows={len(out)}")
    print(out.head(min(12, len(out))).to_string(index=False))


if __name__ == "__main__":
    main()