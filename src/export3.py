import sys, json, pickle; sys.path.insert(0,'.')
from paths import DATA_DIR
import numpy as np, pandas as pd
from model import D, R, FEATS, LAMS, ridge, final_prior, snapshot, FP_FEATS
site = json.load(open(DATA_DIR + '/site_data.json'))
B = site['params']['beta']; H = site['params']['hfa']; MF = ['NT','SR','PTS']
pre = pd.read_pickle(DATA_DIR + '/pregame.pkl').set_index('gid')
def et(ts):
    t = pd.to_datetime(ts, utc=True).tz_convert('America/New_York')
    return t.strftime('%Y-%m-%d'), ('' if (t.hour == 0 and t.minute == 0) else t.strftime('%-I:%M %p'))
games = []
for y in range(2004, 2027):
    sc = pd.read_parquet(f"{DATA_DIR}/sched_{y}.parquet").set_index('game_id')
    played = {}
    for g in D[y][1].sort_values('date').itertuples():
        hp, ap = played.get(g.home_id, 0), played.get(g.away_id, 0)
        mingp = min(hp, ap)
        d, tm = et(sc.loc[g.game_id, 'game_date'])
        pred = None
        if g.game_id in pre.index:
            p = pre.loc[g.game_id]; pred = round(float(sum(B[f]*p[f] for f in MF) + H*p['home']), 2)
        v = None if pd.isna(g.home_spread_margin) else float(g.home_spread_margin)
        games.append([int(g.game_id), d, tm, int(g.home_id), int(g.away_id), bool(g.neutral_site), int(g.home_score), int(g.away_score), pred, v, mingp])
        played[g.home_id], played[g.away_id] = hp + 1, ap + 1
    if y == 2026: played_2026 = played
# upcoming 2026: current ratings, prior on
y = 2026; ids = sorted(D[y][2].team_id); pri = final_prior(y, ids, 0.9); r = R[y]
rate = np.zeros(len(ids))
for f in MF:
    num, wc = FEATS[f]; o, d_, mu, h = ridge(r, ids, num, wc, LAMS[f], pri[f]); rate += B[f]*(o-d_)
rt = dict(zip(ids, rate)); done = set(D[y][1].game_id.astype(int))
up = pd.read_parquet(DATA_DIR + '/up_cfb_schedules_2026.parquet')
n_up = 0
for g in up.itertuples():
    gid = int(g.game_id); hi, ai = int(g.home_id), int(g.away_id)
    if gid in done or hi not in rt or ai not in rt: continue
    d, tm = et(g.start_date)
    if g.start_time_tbd: tm = ''
    pred = round(float(rt[hi] - rt[ai] + (0 if g.neutral_site else H)), 2)
    mingp = min(played_2026.get(hi, 0), played_2026.get(ai, 0))
    games.append([gid, d, tm, hi, ai, bool(g.neutral_site), None, None, pred, None, mingp]); n_up += 1
# pregame projected totals, separate dict so the games row layout (and kalshi.py's padding) stays the same
site['ptot'] = {str(int(gid)): round(float(v), 1) for gid, v in pre['ptot'].dropna().items()} if 'ptot' in pre else {}
num, wc = FEATS['PTS']; po, pdd, pmu, _ = ridge(r, ids, num, wc, LAMS['PTS'], pri['PTS'])
pt = {t: (po[i], pdd[i]) for i, t in enumerate(ids)}
for g in up.itertuples():
    hi, ai = int(g.home_id), int(g.away_id)
    if int(g.game_id) in done or hi not in pt or ai not in pt: continue
    site['ptot'][str(int(g.game_id))] = round(float(2*pmu + pt[hi][0] + pt[ai][1] + pt[ai][0] + pt[hi][1]), 1)
# matchup fingerprints: each side's pregame team snapshot, for Games Like This on the page.
# Past games come from the walk forward (only games before that week); upcoming games from every 2026
# game so far, which is what the walk forward would use for them. Values are deviations from that
# fit's league average, stored as scaled integers to keep the page small; null = not available.
FP_STATS = ["E", "SR", "EX", "PE", "RE", "HV", "FD", "PTS"]
FP_SCALE = {"r": 10, "gp": 1, "sos": 10, "E": 1000, "SR": 10000, "EX": 1000, "PE": 1000, "RE": 1000, "HV": 10000, "FD": 100, "PTS": 10}
fp_cols, fp_scale = [], []
for side in ("h", "a"):
    for c in ["r", "gp", "sos"] + [f"{f}_{od}" for f in FP_STATS for od in ("o", "d")]:
        fp_cols.append(f"{side}_{c}"); fp_scale.append(FP_SCALE[c.split("_")[0]])
def fp_side(get):  # get(feature, 'o'|'d'), gp, sos components -> one side's scaled ints
    q = lambda v, k: None if v is None or not np.isfinite(v) else int(round(v * FP_SCALE[k]))
    r_ = sum(B[f] * (get(f, "o") - get(f, "d")) for f in MF)
    sos = get("sos", None)
    out = [q(r_, "r"), get("gp", None), q(None if sos is None else sum(B[f] * v for f, v in zip(MF, sos)), "sos")]
    for f in FP_STATS:
        out += [q(get(f, "o"), f), q(get(f, "d"), f)]
    return out
fp_rows = []
if "h_E_o" in pre.columns:
    for gid, p in pre.iterrows():
        row = [int(gid)]
        for side in ("h", "a"):
            def get(f, od, p=p, side=side):
                if f == "gp": return int(p[f"{side}_gp"])
                if f == "sos":
                    v = [p[f"{side}_sos_{x}"] for x in MF]
                    return None if any(pd.isna(x) for x in v) else v
                return float(p[f"{side}_{f}_{od}"])
            row += fp_side(get)
        fp_rows.append(row)
    pri_fp = final_prior(y, ids, 0.9, FP_FEATS)
    sn = snapshot(r, ids, pri, pri_fp)
    for g in up.itertuples():
        gid, hi, ai = int(g.game_id), int(g.home_id), int(g.away_id)
        if gid in done or hi not in sn or ai not in sn: continue
        row = [gid]
        for t in (hi, ai):
            s_ = sn[t]
            row += fp_side(lambda f, od, s_=s_: s_["gp"] if f == "gp" else s_["sos"] if f == "sos" else s_[f][0 if od == "o" else 1])
        fp_rows.append(row)
site['fp'] = {"cols": fp_cols, "scale": fp_scale, "rows": fp_rows}
print('fingerprints', len(fp_rows))
games.sort(key=lambda x: (x[1], x[2] == '', pd.to_datetime(x[2], format='%I:%M %p').time() if x[2] else 0))
site['games'] = games; site['asof'] = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')
json.dump(site, open(DATA_DIR + '/site_data.json','w'), separators=(',',':'))
done_g = [g for g in games if g[6] is not None and g[8] is not None]
acc = np.mean([(g[8] > 0) == (g[6] > g[7]) for g in done_g if g[6] != g[7]])
print('games', len(games), 'upcoming', n_up, 'graded', len(done_g), 'SU acc', round(acc,3))
print([g for g in games if g[1] == '2026-10-08'][:3])
