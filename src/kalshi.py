"""Pull live NCAAF moneyline odds from Kalshi and merge them into every upcoming
(unplayed) game in site_data.json, as two more fields per game row: kalshi_home_pct,
kalshi_away_pct (both null if unmatched or the fetch failed). No API key needed for
market data. Run this after export3.py, before build_page.py.
"""
import sys, json, unicodedata
sys.path.insert(0, '.')
from collections import defaultdict
import urllib.request
from paths import DATA_DIR

ALIASES = {
    "appalachian state": "app state", "southern mississippi": "southern miss",
    "hawaii": "hawai i", "louisiana monroe": "ul monroe", "louisiana lafayette": "louisiana",
    "connecticut": "uconn",
}

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", " ").replace("-", " ")
    s = " ".join(s.split())
    if s.endswith(" st"):
        s = s[:-3] + " state"
    return s

def fetch_markets():
    all_markets, cursor = [], None
    while True:
        qs = "series_ticker=KXNCAAFGAME&limit=200" + (f"&cursor={cursor}" if cursor else "")
        req = urllib.request.Request(
            f"https://api.elections.kalshi.com/trade-api/v2/markets?{qs}",
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        all_markets.extend(data["markets"])
        cursor = data.get("cursor")
        if not cursor:
            break
    return [m for m in all_markets if m["status"] == "active"]

def main():
    site = json.load(open(DATA_DIR + '/site_data.json'))
    site['games'] = [g[:11] for g in site['games']]  # idempotent: drop any kalshi fields from a prior run
    our_teams = sorted(set(v[0] for v in site['teams'].values()))
    norm_to_our = {norm(t): t for t in our_teams}
    our_to_id = {v[0]: int(k) for k, v in site['teams'].items()}

    def match_team(kalshi_name):
        n = ALIASES.get(norm(kalshi_name), norm(kalshi_name))
        return norm_to_our.get(n)

    try:
        active = fetch_markets()
    except Exception as e:
        print("Kalshi fetch failed, leaving games unmatched:", e)
        active = []

    by_event = defaultdict(list)
    for m in active:
        by_event[m["event_ticker"]].append(m)

    by_ids = {}  # frozenset of team ids -> game row index, for upcoming games only
    for i, g in enumerate(site['games']):
        if g[6] is not None:  # already played
            continue
        by_ids[frozenset((g[3], g[4]))] = i

    matched = 0
    for event, ms in by_event.items():
        if len(ms) != 2:
            continue
        teams = {}
        ok = True
        for m in ms:
            name = match_team(m["yes_sub_title"])
            if name is None or name not in our_to_id:
                ok = False
                break
            bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
            vol, last = float(m["volume_24h_fp"]), float(m["last_price_dollars"])
            pct = last * 100 if vol > 0 and last > 0 else (bid + ask) / 2 * 100
            teams[name] = round(pct, 1)
        if not ok or len(teams) != 2:
            continue
        ids = frozenset(our_to_id[n] for n in teams)
        idx = by_ids.get(ids)
        if idx is None:
            continue
        g = site['games'][idx]
        home_name = site['teams'][str(g[3])][0]
        away_name = site['teams'][str(g[4])][0]
        g += [teams[home_name], teams[away_name]]
        matched += 1

    # pad every other game row to the same length (14: original 11 + 2 kalshi fields)
    for g in site['games']:
        while len(g) < 13:
            g.append(None)

    print(f"{len(active)} active Kalshi markets, {matched} matched to upcoming games")
    json.dump(site, open(DATA_DIR + '/site_data.json', 'w'), separators=(',', ':'))

if __name__ == "__main__":
    main()
