"""Sportsbook lines from CollegeFootballData's /lines endpoint.

Keeps data/lines_hist.json (committed): {season: {game_id: [close_home_margin, open_home_margin,
total, open_total, home_moneyline, away_moneyline]}}, nulls where a book didn't post one (seasons
cached before moneylines were added have only the first four). Past seasons are fetched once and then
reused; only the current season is refetched each run (2 calls), since CFBD's free tier caps calls.

Then, in site_data.json:
  - upcoming games get their spread (games row index 9, expected home margin), which the
    ESPN data only has for finished games
  - 'lineopen': {gid: opening home margin} for upcoming games
  - 'open': {gid: opening home margin} for every game with an opener (2023 on), for the opening line backtest
  - 'tot': {gid: total} for every game with a posted total, for over/under trends
  - 'ml': {gid: [home American odds, away American odds]} where a book posted a moneyline
Needs CFBD_API_KEY to refresh; without it the cached history is still applied.
CFBD game ids are ESPN ids, same as ours. Run after export3.py, before kalshi.py.
"""
import sys, os, json, urllib.request
sys.path.insert(0, '.')
from paths import DATA_DIR

API_KEY = os.environ.get("CFBD_API_KEY")
SEASON = 2026
FIRST = 2013
HIST = DATA_DIR + '/lines_hist.json'
PROVIDERS = ["consensus", "DraftKings", "ESPN Bet", "Bovada", "William Hill (New Jersey)"]


def cfbd_get(path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"https://api.collegefootballdata.com{path}?{qs}",
                                 headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pick(lines, key):
    """Preferred provider's line that has `key` posted, else the first one that does."""
    usable = [l for l in lines if num(l.get(key)) is not None]
    if not usable:
        return None
    by_prov = {l.get("provider"): l for l in usable}
    for p in PROVIDERS:
        if p in by_prov:
            return by_prov[p]
    return usable[0]


def fetch_season(y):
    out, samples = {}, []
    for st in ("regular", "postseason"):
        for row in cfbd_get("/lines", year=y, seasonType=st):
            lines = row.get("lines") or []
            s, t, m = pick(lines, "spread"), pick(lines, "overUnder"), pick(lines, "homeMoneyline")
            if s is None and t is None and m is None:
                continue
            # CFBD spread is the home side's number (negative = home favored), so home margin is its negative
            close = -num(s["spread"]) if s else None
            opn = -num(s.get("spreadOpen")) if s and num(s.get("spreadOpen")) is not None else None
            tot = num(t["overUnder"]) if t else None
            topn = num(t.get("overUnderOpen")) if t else None
            hml = num(m.get("homeMoneyline")) if m else None
            aml = num(m.get("awayMoneyline")) if m else None
            out[str(row["id"])] = [close, opn, tot, topn, hml, aml]
            if s and len(samples) < 2:
                samples.append((row.get("homeTeam"), row.get("awayTeam"), s.get("spread"), s.get("formattedSpread"), tot))
    return out, samples


def main():
    try:
        hist = json.load(open(HIST))
    except (FileNotFoundError, ValueError):
        hist = {}
    if API_KEY:
        todo = [y for y in range(FIRST, SEASON + 1) if str(y) not in hist or y == SEASON]
        for y in todo:
            try:
                hist[str(y)], samples = fetch_season(y)
                print(f"lines {y}: {len(hist[str(y)])} games", samples if y == SEASON else "")
            except Exception as e:
                print(f"lines fetch failed for {y}:", e)
        json.dump(hist, open(HIST, 'w'), separators=(',', ':'))
    else:
        print("No CFBD_API_KEY set, using cached lines only")

    site = json.load(open(DATA_DIR + '/site_data.json'))
    flat = {int(g): v for season in hist.values() for g, v in season.items()}
    filled = 0
    site['lineopen'], site['tot'], site['ml'], site['open'] = {}, {}, {}, {}
    for g in site['games']:
        v = flat.get(g[0])
        if not v:
            continue
        close, opn, tot = v[0], v[1], v[2]
        if len(v) >= 6 and v[4] is not None and v[5] is not None:
            site['ml'][str(g[0])] = [v[4], v[5]]
        if opn is not None:
            site['open'][str(g[0])] = opn
        if g[6] is None:
            if close is not None:
                g[9] = close
                filled += 1
            if opn is not None:
                site['lineopen'][str(g[0])] = opn
        if tot is not None:
            site['tot'][str(g[0])] = tot
    up = sum(1 for g in site['games'] if g[6] is None)
    print(f"filled spreads for {filled} of {up} upcoming games, {len(site['lineopen'])} with an opening line, "
          f"totals for {len(site['tot'])} games, moneylines for {len(site['ml'])}, openers for {len(site['open'])}")
    json.dump(site, open(DATA_DIR + '/site_data.json', 'w'), separators=(',', ':'))


if __name__ == "__main__":
    main()
