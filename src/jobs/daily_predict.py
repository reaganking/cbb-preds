from __future__ import annotations
import os, sys, subprocess
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
  date date NOT NULL,
  game_id bigint NOT NULL,
  home_team_code text,
  home_team_name text,
  away_team_code text,
  away_team_name text,
  pred_home_margin double precision,
  home_spread double precision,
  prob_home_win double precision,
  home_moneyline_nv double precision,
  away_moneyline_nv double precision,
  PRIMARY KEY (date, game_id)
);
"""
UPSERT_SQL = """
INSERT INTO predictions
(date, game_id, home_team_code, home_team_name, away_team_code, away_team_name,
 pred_home_margin, home_spread, prob_home_win, home_moneyline_nv, away_moneyline_nv)
VALUES
(:date, :game_id, :home_team_code, :home_team_name, :away_team_code, :away_team_name,
 :pred_home_margin, :home_spread, :prob_home_win, :home_moneyline_nv, :away_moneyline_nv)
ON CONFLICT (date, game_id) DO UPDATE SET
  home_team_code=EXCLUDED.home_team_code,
  home_team_name=EXCLUDED.home_team_name,
  away_team_code=EXCLUDED.away_team_code,
  away_team_name=EXCLUDED.away_team_name,
  pred_home_margin=EXCLUDED.pred_home_margin,
  home_spread=EXCLUDED.home_spread,
  prob_home_win=EXCLUDED.prob_home_win,
  home_moneyline_nv=EXCLUDED.home_moneyline_nv,
  away_moneyline_nv=EXCLUDED.away_moneyline_nv;
"""

def season_start_for(d: date) -> date:
    # CBB season starts Nov 1 (previous year if Jan–Oct); fetch only Nov–Apr
    if d.month >= 11:
        return date(d.year, 11, 1)
    else:
        return date(d.year, 11, 1) - timedelta(days=365)

def run(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    today = date.today()
    yday = today - timedelta(days=1)
    start = season_start_for(today)

    # 1) Fetch current season (Nov–Apr) through yesterday
    run([sys.executable, "-m", "src.data.fetch_interstat_range",
         "--start", start.isoformat(), "--end", yday.isoformat(),
         "--out-dir", "data/interstat/daily"])

    # 2) Rebuild games history parquet
    run([sys.executable, "src/pipeline/assemble_history.py"])

    # 3) Predict for today; writes data/interstat/history/board_YYYY-MM-DD.csv
    run([sys.executable, "src/model/predict_for_date.py", today.isoformat()])

    # 4) Upsert into Postgres for the web app to read
    csv_path = f"data/interstat/history/board_{today.isoformat()}.csv"
    df = pd.read_csv(csv_path)

    with engine.begin() as conn:
        conn.exec_driver_sql(CREATE_SQL)
        rows = df.to_dict(orient="records")
        for r in rows:
            # ensure plain python types
            payload = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            # add date in case csv doesn't include it or to be safe
            payload["date"] = today.isoformat()
            conn.execute(text(UPSERT_SQL), payload)
    print(f"Upserted {len(df)} rows for {today.isoformat()}")

if __name__ == "__main__":
    main()
