"""Kalshi forward log: every Kalshi opportunity, decided and frozen at the moment it appeared.

Each run, for every game that hasn't kicked off, each side is checked against the Kalshi tab's rule
(buy price plus taker fee below the books' fair price, price 5 to 95 cents, no more than 12 cents off
fair). The first time a side qualifies, one JSON line is appended to kalshi_flags.jsonl with everything
needed to see why it qualified: both markets' raw bid and ask, which way the buy price was reached, the
books' moneylines or spread, the fair price method and its fitted parameter, the fee, EV, edge, Kelly,
the thresholds, a rule version and the time. Nothing is ever recomputed or rewritten: the file is opened
only to append, existing lines are never touched, and CI refuses to deploy if a committed line changed
(see rebuild.yml). If the rule changes, bump RULE and new lines carry the new version; old ones keep theirs.

The page shows these stored values as they are. Grading adds only facts that came later: the final score
and the last pregame Kalshi buy price (kalshi_history.jsonl), for closing line value.
Run after kalshi.py. Also writes site['kflags'] (every line) into site_data.json.
"""
import sys, os, json, math
sys.path.insert(0, '.')
from datetime import datetime, timezone
import pandas as pd
from paths import DATA_DIR

from forward_log import log_path, provenance, ROOT
LOG = log_path('kalshi_flags.jsonl', 'KFLAG_LOG')  # the real file only in CI; local runs write data/local_kalshi_flags.jsonl
REAL = os.path.join(ROOT, 'kalshi_flags.jsonl')
FORMULA = {"fee": "0.07 * price * (1 - price) per contract (Kalshi taker fee)",
           "fair_ml": "power de-vig: find k with pH^k + pA^k = 1 over the implied chances of the two book moneylines",
           "fair_spread": "1 / (1 + exp(-home margin / K)), K fit by grid on decided games 2008 on",
           "buy": "min(YES ask on this team, 1 - YES bid on the other team)",
           "ev": "fair / (buy + fee) - 1", "kelly_q": "max(0, (fair - cost) / (1 - cost)) / 4",
           "model_p": "normal cdf of model margin / sd (sd_early when a team has no games yet)"}
RULE = "kalshi-ml-v1"
THRESH = {"min_c": 5, "max_c": 95, "max_off_c": 12, "min_ev": 0.0}


def amprob(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def devig_power(a, b):
    """k with a**k + b**k = 1; trims longshots more than favorites. Returns (fair for a, k)."""
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if a ** k + b ** k > 1 else (lo, k)
    k = (lo + hi) / 2
    return a ** k, k


def fit_wink(games):
    """Same grid fit as the page's WINK: home win chance = logistic(spread / K), decided games 2008 on."""
    rows = []
    for g in games:
        if g[6] is None or g[9] is None or g[6] == g[7]:
            continue
        y, m = int(g[1][:4]), int(g[1][5:7])
        if (y if m >= 3 else y - 1) < 2008:
            continue
        rows.append((g[9], 1 if g[6] > g[7] else 0))
    best = (8.3, -math.inf)
    k = 5.0
    while k <= 12.0001:
        ll = sum(math.log(1 / (1 + math.exp(-v / k))) if w else math.log(1 - 1 / (1 + math.exp(-v / k))) for v, w in rows)
        if ll > best[1]:
            best = (round(k, 2), ll)
        k += 0.05
    return best[0]


def kickoff(date, tm):
    return pd.Timestamp(f"{date} {tm or '12:00 PM'}", tz="America/New_York").tz_convert("UTC")


def main():
    site = json.load(open(DATA_DIR + '/site_data.json'))
    open(LOG, 'a').close()
    seen, lines = set(), []
    for path in dict.fromkeys([REAL, LOG]):  # read the real log (and a local one when not in CI), write only LOG
        if not os.path.exists(path):
            continue
        for ln in open(path):
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if (e["gid"], e["side"]) not in seen:
                    lines.append(e); seen.add((e["gid"], e["side"]))
    Q, ML = site.get('kquote', {}), site.get('ml', {})
    now = pd.Timestamp(datetime.now(timezone.utc))
    wink = None
    new = []
    for g in site['games']:
        gid = g[0]
        q = Q.get(str(gid))
        if g[6] is not None or not q or kickoff(g[1], g[2]) <= now:
            continue
        ml = ML.get(str(gid))
        if ml and ml[0] is not None and ml[1] is not None:
            pH, k = devig_power(amprob(ml[0]), amprob(ml[1])); method, param = "books_ml_power", {"k": round(k, 5)}
        elif g[9] is not None:
            wink = wink or fit_wink(site['games'])
            pH = 1 / (1 + math.exp(-g[9] / wink)); method, param = "spread_logistic", {"wink": wink}
        else:
            continue
        for side, team, opp, p in (("home", g[3], g[4], pH), ("away", g[4], g[3], 1 - pH)):
            if (gid, side) in seen:
                continue
            me, other = q[side], q["away" if side == "home" else "home"]
            opts = []
            if 0 < me["ask"] < 1:
                opts.append((me["ask"], "yes_ask"))
            if 0 < other["bid"] < 1:
                opts.append((1 - other["bid"], "no_on_other_at_1_minus_bid"))
            if not opts:
                continue
            c, via = min(opts)
            c = round(c, 4)
            fee = 0.07 * c * (1 - c)
            cost = c + fee
            ev = p / cost - 1
            edge = (p - c) * 100
            if not (ev > THRESH["min_ev"] and THRESH["min_c"] <= c * 100 <= THRESH["max_c"] and abs(edge) <= THRESH["max_off_c"]):
                continue
            sd = site['params']['sd_early'] if g[10] == 0 else site['params']['sd']
            mp = None if g[8] is None else 0.5 * (1 + math.erf(g[8] / sd / math.sqrt(2)))
            e = {"v": 2, "rule": RULE, "ts": now.strftime('%Y-%m-%dT%H:%M:%SZ'), "gid": gid, "date": g[1], "time": g[2],
                 "kickoff_utc": kickoff(g[1], g[2]).strftime('%Y-%m-%dT%H:%MZ'), "home": g[3], "away": g[4],
                 "side": side, "team": team, "opp": opp, "buy": c, "buy_via": via, "quotes": q,
                 "fair_method": method, "fair_param": param, "books_ml": ml if ml and ml[0] is not None and ml[1] is not None else None, "spread": g[9],
                 "open_spread": site.get('lineopen', {}).get(str(gid)),
                 "fair": round(p, 5), "fee": round(fee, 5), "cost": round(cost, 5), "ev": round(ev, 5), "edge_c": round(edge, 2),
                 "kelly_q": round(max(0.0, (p - cost) / (1 - cost)) / 4, 5), "thresholds": THRESH,
                 "model_pred": g[8], "model_p_side": None if mp is None else round(mp if side == "home" else 1 - mp, 5),
                 "model_note": "context only, not used to qualify", "formula": FORMULA, **provenance()}
            new.append(e); seen.add((gid, side))
    if new:
        with open(LOG, 'a') as f:  # append only: existing lines are never rewritten
            for e in new:
                f.write(json.dumps(e, separators=(',', ':')) + "\n")
    lines += new
    site['kflags'] = lines
    print(f"kalshi flags: {len(new)} new, {len(lines)} total")
    json.dump(site, open(DATA_DIR + '/site_data.json', 'w'), separators=(',', ':'))


if __name__ == "__main__":
    main()
