import argparse, os, subprocess, sys
from datetime import datetime, timedelta, date

def _daterange(d0: date, d1: date):
    cur = d0
    one = timedelta(days=1)
    while cur <= d1:
        yield cur
        cur += one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/interstat/daily")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()

    ok = fail = 0
    for d in _daterange(start, end):
        out_path = os.path.join(args.out_dir, f"{d.isoformat()}.parquet")
        if args.skip_existing and os.path.exists(out_path):
            print(f"Skip {d.isoformat()} (exists)")
            ok += 1
            continue
        cmd = [sys.executable, "-m", "src.data.fetch_interstat", d.isoformat(), "--out-dir", args.out_dir]
        if args.debug:
            cmd.append("--debug")
        print(" ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
    print(f"Done. Success={ok}, Failures={fail}")

if __name__ == "__main__":
    main()
