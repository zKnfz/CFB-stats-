"""Forward test of the model against the betting line, logged as it happens.

The backtest says the model's side covered the opener about 54.6% in 2025+ games, but it can't rule out
a timing problem: some openers are posted before the previous week's games, which the backtest's model
had already seen (with the model one game older, the cover rate falls to about 51%). This log can't have
that problem. Every run, each game in the next 8 days that has a posted line and isn't logged yet gets one
line in tracking.jsonl (committed, append only): the line at that moment, the model's number at that same
moment, and the opener for reference. Since runs are every 3 hours, a newly posted line is logged within
about 3 hours of going up. Entries are never edited. The page grades the model's side against the logged
line and the closing line once games are played.

Also writes site['track'] (every logged entry) into site_data.json. Run after lines.py, before build_page.py.
"""
import sys, os, json
from datetime import datetime, timezone
sys.path.insert(0, '.')
from paths import DATA_DIR
import pandas as pd

LOG = os.environ.get('TRACK_LOG') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking.jsonl')  # override for local tests
FIELDS = "gid date home away open model line logged lean rule".split()  # open may be null; line is what gets graded
WINDOW_DAYS = 8
RULE, MIN_EDGE = "spread-v1", 1.0  # a lean is the model's side when it disagrees with the logged line by 1+ points


def kickoff_utc(date, tm):
    """ET date and time string from the games row, as UTC; a game with no time counts as noon ET."""
    ts = pd.Timestamp(f"{date} {tm or '12:00 PM'}", tz="America/New_York")
    return ts.tz_convert("UTC")


def main():
    site = json.load(open(DATA_DIR + '/site_data.json'))
    open(LOG, 'a').close()  # always exists, so the bot's git add never fails
    seen = {}
    if os.path.exists(LOG):
        for ln in open(LOG):
            ln = ln.strip()
            if ln:
                e = json.loads(ln); seen.setdefault(e["gid"], e)  # union merges can repeat a line; first one wins
    now = pd.Timestamp(datetime.now(timezone.utc))
    opens = site.get('lineopen', {})
    new = []
    for g in site['games']:
        gid, date, tm, h, a, hs, pred, spread = g[0], g[1], g[2], g[3], g[4], g[6], g[8], g[9]
        op = opens.get(str(gid))
        if gid in seen or hs is not None or spread is None or pred is None:
            continue
        ko = kickoff_utc(date, tm)
        if ko <= now or ko > now + pd.Timedelta(days=WINDOW_DAYS):  # before kickoff, and only this coming week
            continue
        edge = pred - spread  # frozen with the entry, so changing the rule later can't rewrite which side was taken
        e = {"gid": gid, "date": date, "home": h, "away": a, "open": op, "model": pred, "line": spread,
             "logged": now.strftime('%Y-%m-%dT%H:%MZ'), "rule": RULE, "min_edge": MIN_EDGE, "edge": round(edge, 2),
             "lean": ("home" if edge > 0 else "away") if abs(edge) >= MIN_EDGE else None}
        new.append(e); seen[gid] = e
    if new:
        with open(LOG, 'a') as f:
            for e in new:
                f.write(json.dumps(e, separators=(',', ':')) + "\n")
    site['track'] = [[e.get(k) for k in FIELDS] for e in sorted(seen.values(), key=lambda e: (e["date"], e["gid"]))]
    print(f"tracking: {len(new)} new games logged, {len(seen)} total")
    json.dump(site, open(DATA_DIR + '/site_data.json', 'w'), separators=(',', ':'))


if __name__ == "__main__":
    main()
