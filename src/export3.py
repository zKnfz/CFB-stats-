import sys, json, pickle; sys.path.insert(0,'.')
from paths import DATA_DIR
import numpy as np, pandas as pd
from model import D, R, FEATS, LAMS, ridge, final_prior
site = json.load(open(DATA_DIR + '/site_data.json'))
B = site['params']['beta']; H = site['params']['hfa']; MF = ['NT','SR','PTS']
pre = pd.read_pickle(DATA_DIR + '/pregame.pkl').set_index('gid')
def et(ts):
    t = pd.to_datetime(ts, utc=True).tz_convert('America/New_York')
    return t.strftime('%Y-%m-%d'), ('' if (t.hour == 0 and t.minute == 0) else t.strftime('%-I:%M %p'))
games = []
for y in range(2004, 2027):
    sc = pd.read_parquet(f"{DATA_DIR}/sched_{y}.parquet").set_index('game_id')
    for g in D[y][1].itertuples():
        d, tm = et(sc.loc[g.game_id, 'game_date'])
        pred = None
        if g.game_id in pre.index:
            p = pre.loc[g.game_id]; pred = round(float(sum(B[f]*p[f] for f in MF) + H*p['home']), 2)
        v = None if pd.isna(g.home_spread_margin) else float(g.home_spread_margin)
        games.append([int(g.game_id), d, tm, int(g.home_id), int(g.away_id), bool(g.neutral_site), int(g.home_score), int(g.away_score), pred, v])
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
    games.append([gid, d, tm, hi, ai, bool(g.neutral_site), None, None, pred, None]); n_up += 1
games.sort(key=lambda x: (x[1], x[2] == '', pd.to_datetime(x[2], format='%I:%M %p').time() if x[2] else 0))
site['games'] = games; site['asof'] = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')
json.dump(site, open(DATA_DIR + '/site_data.json','w'), separators=(',',':'))
done_g = [g for g in games if g[6] is not None and g[8] is not None]
acc = np.mean([(g[8] > 0) == (g[6] > g[7]) for g in done_g if g[6] != g[7]])
print('games', len(games), 'upcoming', n_up, 'graded', len(done_g), 'SU acc', round(acc,3))
print([g for g in games if g[1] == '2026-10-08'][:3])
