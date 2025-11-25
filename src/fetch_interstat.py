# src/data/fetch_interstat.py
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any

import requests
import pandas as pd


def _to_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def fetch_day(date_str: str, timeout: float = 15.0) -> pd.DataFrame:
    """
    Pull Interstat day scoreboard and return TWO rows per game (one per team).
    Columns are intentionally simple so we can build history cleanly.
    """
    url = f"https://interst.at/game/mbb/{date_str}"
    hdrs = {"Accept": "application/json, */*"}
    r = requests.get(url, headers=hdrs, timeout=timeout)
    r.raise_for_status()

    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"Non-JSON response from {url[:80]}...")

    games: Dict[str, Any] = data.get("games", {})
    rows: List[Dict[str, Any]] = []

    for gkey, g in games.items():
        game_id = g.get("id")
        gameday = g.get("gameday") or date_str
        status = g.get("status")
        starttime = g.get("starttime")
        score = g.get("score", {}) or {}
        overtime = score.get("overtime")
        venue = g.get("venue", {}) or {}
        attendance = g.get("attendance")
        meta = g.get("meta", {}) or {}

        v = g.get("visitor", {}) or {}
        h = g.get("home", {}) or {}

        v_id = v.get("id")
        h_id = h.get("id")

        v_code = (v.get("code") or "") if isinstance(v.get("code"), str) else ""
        h_code = (h.get("code") or "") if isinstance(h.get("code"), str) else ""

        v_team = (v.get("team_fullname") or v.get("team") or "")
        h_team = (h.get("team_fullname") or h.get("team") or "")

        v_pts = _to_int(v.get("score"), default=None)
        h_pts = _to_int(h.get("score"), default=None)

        # Build one row for VIS, one for HOME
        base = dict(
            date=gameday,
            game_id=_to_int(game_id, default=None),
            status=status,
            start_time=starttime,
            overtime=overtime,
            venue_id=_to_int(venue.get("id"), default=None),
            venue_name=venue.get("name"),
            citystate=venue.get("citystate"),
            neutral=(venue.get("neutral") == "Y"),
            attendance=_to_int(attendance, default=None),
            pbp_count=_to_int(meta.get("playbyplay"), default=None),
            playerstatlines_count=_to_int(meta.get("playerstatlines"), default=None),
            siteurl=meta.get("siteurl"),
            apiurl=meta.get("apiurl"),
        )

        # Visitor row
        rows.append(
            dict(
                **base,
                is_home=False,
                team_id=_to_int(v_id, default=None),
                team_code=v_code,
                team_name=v_team,
                opp_id=_to_int(h_id, default=None),
                opp_code=h_code,
                opp_name=h_team,
                pts=v_pts,
                opp_pts=h_pts,
                margin=(v_pts - h_pts) if v_pts is not None and h_pts is not None else None,
            )
        )

        # Home row
        rows.append(
            dict(
                **base,
                is_home=True,
                team_id=_to_int(h_id, default=None),
                team_code=h_code,
                team_name=h_team,
                opp_id=_to_int(v_id, default=None),
                opp_code=v_code,
                opp_name=v_team,
                pts=h_pts,
                opp_pts=v_pts,
                margin=(h_pts - v_pts) if v_pts is not None and h_pts is not None else None,
            )
        )

    df = pd.DataFrame(rows)

    # Ensure types on key fields
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for col in ["game_id", "team_id", "opp_id", "attendance", "pbp_count", "playerstatlines_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        for col in ["pts", "opp_pts", "margin"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "is_home" in df.columns:
            df["is_home"] = df["is_home"].astype(bool)

    return df


def save_daily(df: pd.DataFrame, out_dir: str, date_str: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.parquet")
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        # Fallback to CSV if pyarrow/fastparquet missing
        out_path = os.path.join(out_dir, f"{date_str}.csv")
        df.to_csv(out_path, index=False)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Fetch Interstat MBB day scoreboard -> two rows per game (team-level).")
    parser.add_argument("date", help="YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/interstat/daily", help="Output directory (default: data/interstat/daily)")
    parser.add_argument("--debug", action="store_true", help="Print shape/head after fetch")
    args = parser.parse_args()

    # Validate date early
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("date must be YYYY-MM-DD")

    df = fetch_day(args.date)
    if df.empty:
        print(f"[{args.date}] No games found or empty payload.")
        return

    out_path = save_daily(df, args.out_dir, args.date)
    print(f"Saved {len(df)} rows -> {out_path}")

    if args.debug:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.head(10))


if __name__ == "__main__":
    main()