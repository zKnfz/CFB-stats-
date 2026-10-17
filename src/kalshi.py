"""Pull live NCAAF moneyline odds from Kalshi and merge them into every upcoming
(unplayed) game in site_data.json, as two more fields per game row: kalshi_home_pct,
kalshi_away_pct (both null if unmatched or the fetch failed; a midpoint style price for display).
Also 'kbuy': {gid: [home buy, away buy, home 24h volume, away 24h volume]}: the cheapest price to
actually get each team, in cents. Buying a team is either YES on its market at the ask or NO on the
other team's market at 100 minus that market's bid, whichever is cheaper. EV on the page uses these.
Also logs which college football series Kalshi lists (probe), so spread and total markets can be
added once their tickers and format are known. No API key needed for market data. Run after export3.py.
"""
import sys, json, unicodedata, os, datetime, time
sys.path.insert(0, '.')
from collections import defaultdict
import urllib.request
from paths import DATA_DIR

HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kalshi_history.jsonl')

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



def kickoff_passed(date, tm):
    """Games rows carry an ET date and time ('7:30 PM', or '' when not set, counted as noon ET)."""
    import pandas as pd
    ko = pd.Timestamp(f"{date} {tm or '12:00 PM'}", tz="America/New_York")
    return ko <= pd.Timestamp.now(tz="UTC")


def fetch_markets(series="KXNCAAFGAME"):
    all_markets, cursor = [], None
    while True:
        qs = f"series_ticker={series}&limit=200" + (f"&cursor={cursor}" if cursor else "")
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

def probe():
    """Which college football series does Kalshi list? One request to the series list, logs the matches, changes nothing.
    (Guessing KXNCAAFSPREAD / KXNCAAFTOTAL got HTTP 429 every time while the winner series worked, so ask instead.)"""
    try:
        req = urllib.request.Request("https://api.elections.kalshi.com/trade-api/v2/series?category=Sports",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            series = json.loads(r.read()).get("series") or []
        hits = [x for x in series if "NCAAF" in (x.get("ticker") or "").upper() or "college football" in (x.get("title") or "").lower()]
        print(f"probe: {len(series)} sports series, {len(hits)} college football:")
        for x in hits[:40]:
            print("  ", x.get("ticker"), "|", x.get("title"), "|", x.get("frequency"))
    except Exception as e:
        print("probe failed:", e)


def main():
    probe()
    site = json.load(open(DATA_DIR + '/site_data.json'))
    site['games'] = [g[:11] for g in site['games']]  # idempotent: drop any kalshi fields from a prior run
    site['kbuy'] = {}
    site['kquote'] = {}
    our_teams = sorted(set(v[0] for v in site['teams'].values()))
    norm_to_our = {norm(t): t for t in our_teams}
    our_to_id = {v[0]: int(k) for k, v in site['teams'].items()}

    def match_team(kalshi_name):
        n = ALIASES.get(norm(kalshi_name), norm(kalshi_name))
        return norm_to_our.get(n)

    active = []
    for wait in (5, 20, 45):  # the probe just ran; Kalshi answers bursts with HTTP 429, so pause and retry
        time.sleep(wait)
        try:
            active = fetch_markets(); break
        except Exception as e:
            print("Kalshi fetch failed:", e)
    if not active:
        print("Kalshi fetch gave nothing, leaving games unmatched")

    by_event = defaultdict(list)
    for m in active:
        by_event[m["event_ticker"]].append(m)

    by_ids = {}  # frozenset of team ids -> game row index, for upcoming games only
    for i, g in enumerate(site['games']):
        if g[6] is not None:  # already played
            continue
        by_ids[frozenset((g[3], g[4]))] = i

    matched = 0
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    snapshot_lines = []
    for event, ms in by_event.items():
        if len(ms) != 2:
            continue
        teams = {}
        vols = {}
        quotes = {}
        tickers = {}
        ok = True
        for m in ms:
            name = match_team(m["yes_sub_title"])
            if name is None or name not in our_to_id:
                ok = False
                break
            bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
            vol, last = float(m["volume_24h_fp"]), float(m["last_price_dollars"])
            # a live two sided quote beats the last trade, which can be hours stale on thin markets
            pct = (bid + ask) / 2 * 100 if bid > 0 and ask < 1 and ask - bid <= 0.10 else (last * 100 if vol > 0 and last > 0 else (bid + ask) / 2 * 100)
            teams[name] = round(pct, 1)
            vols[name] = vol
            quotes[name] = (bid, ask)
            tickers[name] = m.get("ticker")
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
        # cheapest way to own each side: YES on it at the ask, or NO on the other side at 1 minus its bid
        def buy(me, other):
            opts = [q for q in (quotes[me][1], 1 - quotes[other][0] if quotes[other][0] > 0 else None) if q is not None and 0 < q < 1]
            return round(min(opts) * 100, 1) if opts else None
        site.setdefault('kbuy', {})[str(g[0])] = [buy(home_name, away_name), buy(away_name, home_name), vols[home_name], vols[away_name]]
        # raw quotes as Kalshi showed them, for kalshi_flags.py to freeze with each opportunity
        site.setdefault('kquote', {})[str(g[0])] = {side: {"bid": quotes[n][0], "ask": quotes[n][1], "ticker": tickers[n], "vol24h": vols[n]}
                                                  for side, n in (("home", home_name), ("away", away_name))}
        # log a snapshot now, so line movement over time becomes buildable going forward,
        # even without historical opening lines. Pregame only: a game that kicked off but isn't final
        # still counts as upcoming, and in game prices would spoil the price history and the track record.
        if kickoff_passed(g[1], g[2]):
            continue
        snapshot_lines.append(json.dumps({
            "ts": now, "gid": g[0], "date": g[1], "home_id": g[3], "away_id": g[4],
            "home_pct": teams[home_name], "away_pct": teams[away_name],
            "home_vol24h": vols[home_name], "away_vol24h": vols[away_name],
            "model_pred": g[8], "vegas_line": g[9],
            # for the Kalshi track record: what you could buy each side for, and the books' moneylines, at this moment
            "home_buy": site['kbuy'][str(g[0])][0], "away_buy": site['kbuy'][str(g[0])][1],
            "home_ml": (site.get('ml') or {}).get(str(g[0]), [None, None])[0], "away_ml": (site.get('ml') or {}).get(str(g[0]), [None, None])[1],
        }))

    if snapshot_lines:
        with open(HISTORY_PATH, 'a') as f:
            f.write('\n'.join(snapshot_lines) + '\n')

    # price history for games still upcoming, for the Trends tab's line movement sparkline
    up_ids = {g[0] for g in site['games'] if g[6] is None}
    moves = defaultdict(list)
    try:
        with open(HISTORY_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("gid") in up_ids and r.get("home_pct") is not None:
                    moves[r["gid"]].append([r["ts"], r["home_pct"]])
    except FileNotFoundError:
        pass
    site['kmove'] = {str(k): v[-40:] for k, v in moves.items()}

    # every snapshot with buy prices (logged since the Kalshi track record started), for every game, compact:
    # {gid: [[ts, home buy, away buy, home ml, away ml, spread, home mid], ...]}, repeated identical snapshots dropped.
    # The page finds the first snapshot where Kalshi was cheaper than the books' fair price and grades it.
    snaps = defaultdict(list)
    try:
        with open(HISTORY_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "home_buy" not in r:
                    continue
                row = [r["ts"][:16], r["home_buy"], r["away_buy"], r.get("home_ml"), r.get("away_ml"), r.get("vegas_line"), r.get("home_pct")]
                prev = snaps[r["gid"]]
                if not prev or prev[-1][1:] != row[1:]:
                    prev.append(row)
    except FileNotFoundError:
        pass
    site['ksnap'] = {str(k): v for k, v in snaps.items()}

    # pad every other game row to the same length (14: original 11 + 2 kalshi fields)
    for g in site['games']:
        while len(g) < 13:
            g.append(None)

    print(f"{len(active)} active Kalshi markets, {matched} matched to upcoming games")
    json.dump(site, open(DATA_DIR + '/site_data.json', 'w'), separators=(',', ':'))

if __name__ == "__main__":
    main()
