import sys, json; sys.path.insert(0,'.')
from paths import DATA_DIR
import numpy as np, pandas as pd
from model import D, R, FEATS, LAMS, ridge, final_prior
from scipy.stats import norm
SHRINK = 0.9; MF = ['NT','SR','PTS']
# bump this whenever FEATS, MF, beta, a lambda, or SHRINK changes, so old backtest rows
# can be told apart from a different model's. Not bumped for display-only additions.
MODEL_VERSION = "1.0.0"
bt = pd.read_pickle(DATA_DIR + '/bt.pkl'); bt = bt[(bt.shrink==0.9)&(bt.hl==0)&(bt.year!=2020)]
tr = bt[bt.year<=2024]
X = np.c_[tr[MF], tr.home]; beta = np.linalg.lstsq(X, tr.margin, rcond=None)[0]
pred = lambda d: np.c_[d[MF], d.home] @ beta
sd = float((pred(tr)-tr.margin).std())
# early-season games (neither team has played yet this season) miss by more: widen the band just for those
counts = []
for y in sorted(D.keys()):
    played = {}
    for g in D[y][1].sort_values('date').itertuples():
        hp, ap = played.get(g.home_id, 0), played.get(g.away_id, 0)
        counts.append((y, g.game_id, min(hp, ap)))
        played[g.home_id], played[g.away_id] = hp + 1, ap + 1
mingp = pd.DataFrame(counts, columns=['year', 'gid', 'mingp'])
tr_m = tr.merge(mingp, on=['year', 'gid'], how='left')
early = tr_m[tr_m.mingp == 0]
sd_early = float((pred(early) - early.margin).std())
def sc(d):
    v = d.dropna(subset=['vegas']); p = pred(d)
    return {"n": len(d), "mae": float((p-d.margin).abs().mean()), "acc": float((np.sign(p)==np.sign(d.margin)).mean()),
            "vegas_mae": float((v.vegas-v.margin).abs().mean()), "vegas_acc": float((np.sign(v.vegas)==np.sign(v.margin)).mean())}
old = lambda d: d  # for reporting the old model on same test set
te = bt[bt.year>=2025]
Xo = np.c_[tr.E, tr.home]; bo = np.linalg.lstsq(Xo, tr.margin, rcond=None)[0]
old_mae = float((np.c_[te.E, te.home]@bo - te.margin).abs().mean())
bt_out = {"test": sc(te), "test_old_mae": old_mae, "val": sc(bt[(bt.year>=2021)&(bt.year<=2024)]),
          "train_years": "2008 to 2024", "test_years": "2025 and 2026 so far"}
print('beta', beta, 'sd', sd, bt_out)
teams_all = pd.concat([D[y][2] for y in sorted(D)]).drop_duplicates('team_id', keep='last')
out = {"params": {"lam": LAMS, "beta": {f: round(float(b),4) for f,b in zip(MF, beta[:3])}, "hfa": round(float(beta[3]),3), "sd": round(sd,3), "sd_early": round(sd_early,3), "shrink": SHRINK, "model_version": MODEL_VERSION},
       "backtest": bt_out, "teams": {int(r.team_id): [r.school, r.abbreviation, r.conference, r.color] for r in teams_all.itertuples()},
       "seasons": {}, "coverage": {}}
for y in range(2004, 2027):
    sc_ = pd.read_parquet(f"{DATA_DIR}/sched_{y}.parquet"); fbs = set(D[y][2].team_id)
    sc_ = sc_[sc_.home_id.astype(int).isin(fbs) & sc_.away_id.astype(int).isin(fbs)]
    out["coverage"][str(y)] = round(len(D[y][1])/max(len(sc_),1), 3)
    rows, games, teams = R[y], D[y][1], D[y][2]
    ids = sorted(teams.team_id)
    pri = final_prior(y, ids, SHRINK)
    n = len(ids)
    prior = {} if pri is None else {int(t): {f: [round(float(pri[f][i]),4), round(float(pri[f][n+i]),4)] for f in FEATS} for i,t in enumerate(ids)}
    Rr = rows.sort_values(['date','game_id'])
    cols = ['date','game_id','off','def','loc','plays','epa','succ','pplays','pepa','tov','epa_nt','plays_nt','pts','ex_nt','plays_ex']
    out["seasons"][y] = {"teams": [int(t) for t in ids], "prior": prior,
        "conf": {int(r.team_id): r.conference for r in teams.itertuples()},
        "rows": [[a,int(b),int(c),int(d_),int(e),int(f),round(float(g),3),int(h),int(i),round(float(j),3),int(k),round(float(l),3),int(m),int(nn),round(float(oo),3),int(pp)]
                 for a,b,c,d_,e,f,g,h,i,j,k,l,m,nn,oo,pp in Rr[cols].itertuples(index=False)],
        "games": {int(g.game_id): [g.date,int(g.home_id),int(g.away_id),int(g.home_score),int(g.away_score),bool(g.neutral_site),int(g.season_type),int(g.week)] for g in games.itertuples()}}
json.dump(out, open(DATA_DIR + '/site_data.json','w'), separators=(',',':'))
# python reference for 2026 with prior, to check the browser
y=2026; ids=sorted(D[y][2].team_id); pri=final_prior(y, ids, SHRINK); r=R[y]
tot=np.zeros(len(ids))
for f,b in zip(MF,beta[:3]):
    num,wc=FEATS[f]; o,d,mu,h=ridge(r,ids,num,wc,LAMS[f],pri[f]); tot+=b*(o-d)
nm=dict(zip(D[y][2].team_id,D[y][2].school)); top=np.argsort(-tot)[:6]
print('REF', [(nm[ids[i]], round(tot[i],2)) for i in top])
import os; print(os.path.getsize(DATA_DIR + '/site_data.json'))
