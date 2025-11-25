from __future__ import annotations
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import date
import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="CBB Board")

TABLE_SQL = """
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
with engine.begin() as conn:
    conn.exec_driver_sql(TABLE_SQL)

@app.get("/api/predictions", response_class=JSONResponse)
def api_predictions(d: str | None = Query(default=None)):
    d = d or date.today().isoformat()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""SELECT * FROM predictions WHERE date = :d ORDER BY home_spread ASC, game_id"""),
            {"d": d}
        ).mappings().all()
    return {"date": d, "rows": [dict(r) for r in rows]}

@app.get("/", response_class=HTMLResponse)
def index(d: str | None = Query(default=None)):
    d = d or date.today().isoformat()
    with engine.begin() as conn:
        df = pd.read_sql(
            text("""SELECT * FROM predictions WHERE date = :d ORDER BY home_spread ASC, game_id"""),
            conn, params={"d": d}
        )
    title = f"CBB Board — {d}"
    if df.empty:
        body = f"<p>No predictions yet for <b>{d}</b>.</p><p>Check back after the 9:05am run.</p>"
    else:
        # nicify columns
        show = df[[
            "home_team_code","home_team_name","away_team_code","away_team_name",
            "home_spread","prob_home_win","home_moneyline_nv","away_moneyline_nv","pred_home_margin"
        ]].copy()
        show.rename(columns={
            "home_team_code":"HOME",
            "home_team_name":"Home Team",
            "away_team_code":"AWAY",
            "away_team_name":"Away Team",
            "home_spread":"Spread (Home -)",
            "prob_home_win":"Home Win %",
            "home_moneyline_nv":"Home ML",
            "away_moneyline_nv":"Away ML",
            "pred_home_margin":"Pred Margin"
        }, inplace=True)
        # formatting
        show["Home Win %"] = (show["Home Win %"]*100).round(1)
        show["Spread (Home -)"] = show["Spread (Home -)"].round(1)
        show["Pred Margin"] = show["Pred Margin"].round(1)
        show["Home ML"] = show["Home ML"].round(0).astype("Int64").astype(str).str.replace("<NA>","")
        show["Away ML"] = show["Away ML"].round(0).astype("Int64").astype(str).str.replace("<NA>","")
        # simple HTML table
        body = show.to_html(index=False, escape=False)

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    form {{ margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; }}
    th {{ background: #f3f4f6; text-align: left; }}
    tr:nth-child(even) td {{ background: #fafafa; }}
    .hint {{ color:#6b7280; font-size: 13px; }}
    .datebox {{ padding:6px 8px; border:1px solid #d1d5db; border-radius:6px; }}
    .btn {{ padding:6px 10px; border:1px solid #111827; background:#111827; color:white; border-radius:6px; cursor:pointer; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <form method="get" action="/">
    <input class="datebox" type="date" name="d" value="{d}">
    <button class="btn" type="submit">Go</button>
    <span class="hint">Use the date picker to view a different day.</span>
  </form>
  {body}
</body>
</html>
"""
    return HTMLResponse(html)
