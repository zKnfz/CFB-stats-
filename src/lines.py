"""Fill in sportsbook spreads for upcoming games from CollegeFootballData's /lines endpoint.
Finished games already carry a closing line from the ESPN data; upcoming ones have none,
so their vegas field (games row index 9, expected home margin) is null until this runs.
Needs CFBD_API_KEY, skips gracefully without it. Run after export3.py, before kalshi.py.
CFBD game ids are ESPN ids, same as ours, so games match by id directly.
"""
import sys, os, json, urllib.request
sys.path.insert(0, '.')
from paths import DATA_DIR

API_KEY = os.environ.get("CFBD_API_KEY")
SEASON = 2026
PROVIDERS = ["consensus", "DraftKings", "ESPN Bet", "Bovada", "William Hill (New Jersey)"]


def cfbd_get(path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"https://api.collegefootballdata.com{path}?{qs}",
                                 headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def pick_line(lines):
    """Preferred provider's spread (home perspective, negative = home favored), else the first one posted."""
    usable = [l for l in lines if l.get("spread") not in (None, "")]
    if not usable:
        return None
    by_prov = {l.get("provider"): l for l in usable}
    for p in PROVIDERS:
        if p in by_prov:
            return by_prov[p]
    return usable[0]


def main():
    if not API_KEY:
        print("No CFBD_API_KEY set, skipping lines fetch")
        return
    site = json.load(open(DATA_DIR + '/site_data.json'))
    upcoming = {g[0]: g for g in site['games'] if g[6] is None}

    spreads, opens, samples = {}, {}, []
    for st in ("regular", "postseason"):
        try:
            rows = cfbd_get("/lines", year=SEASON, seasonType=st)
        except Exception as e:
            print(f"lines fetch failed for {st}:", e)
            continue
        for row in rows:
            l = pick_line(row.get("lines") or [])
            if l is None:
                continue
            gid = int(row["id"])
            spreads[gid] = -float(l["spread"])  # CFBD spread is the home side's number, so home margin is its negative
            if l.get("spreadOpen") not in (None, ""):
                opens[gid] = -float(l["spreadOpen"])
            if gid in upcoming and len(samples) < 3:
                samples.append((row.get("homeTeam"), row.get("awayTeam"), l.get("provider"), l.get("spread"), l.get("formattedSpread")))

    filled = 0
    site['lineopen'] = {}
    for gid, g in upcoming.items():
        if gid in spreads:
            g[9] = spreads[gid]
            filled += 1
        if gid in opens:
            site['lineopen'][str(gid)] = opens[gid]
    print(f"{len(spreads)} games with lines from CFBD, filled {filled} of {len(upcoming)} upcoming games, {len(site['lineopen'])} with an opening line")
    print("samples (home, away, provider, spread, formatted):", samples)
    json.dump(site, open(DATA_DIR + '/site_data.json', 'w'), separators=(',', ':'))


if __name__ == "__main__":
    main()
