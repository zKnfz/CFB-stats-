"""Walk-forward feature generation for model selection (vectorized ridge)."""
from paths import DATA_DIR
import pickle
import numpy as np, pandas as pd

D = pickle.load(open(DATA_DIR + "/prep.pkl", "rb"))
FEATS = {  # name: (numerator col, weight col)
    "E": ("epa", "plays"), "NT": ("epa_nt", "plays_nt"), "SR": ("succ", "plays"), "PTS": ("pts", "one"),
}


def prep_rows(rows):
    r = rows.copy()
    r["pts"] = np.where(r.off == r.home_id, r.home_score, r.away_score).astype(float)
    r["one"] = 1.0
    r["dt"] = pd.to_datetime(r.date)
    return r


R = {y: prep_rows(D[y][0]) for y in D}


def ridge(r, ids, num, wcol, lam, prior=None, decay_w=None):
    n = len(ids); idx = {t: i for i, t in enumerate(ids)}
    oi = r.off.map(idx).to_numpy(); di = r["def"].map(idx).to_numpy()
    w = r[wcol].to_numpy(float) * (1 if decay_w is None else decay_w)
    y = r[num].to_numpy(float) / np.maximum(r[wcol].to_numpy(float), 1e-9)
    m = 2 * n + 2
    X = np.zeros((len(r), m)); ar = np.arange(len(r))
    X[ar, oi] = 1; X[ar, n + di] = 1; X[:, 2 * n] = 1; X[:, 2 * n + 1] = r["loc"].to_numpy(float)
    A = X.T @ (X * w[:, None]); b = X.T @ (w * y)
    pr = np.zeros(2 * n) if prior is None else prior
    A[np.arange(2 * n), np.arange(2 * n)] += lam; b[:2 * n] += lam * pr
    A[2 * n, 2 * n] += 1e-6; A[2 * n + 1, 2 * n + 1] += 1e-6
    x = np.linalg.solve(A, b)
    return x[:n], x[n:2 * n], x[2 * n], x[2 * n + 1]


LAMS = {"E": 100, "NT": 100, "SR": 100, "PTS": 3}


def final_prior(year, ids, shrink):
    """Last season's final o,d per feature, shrunk; missing teams get a below-average default."""
    out = {}
    if year - 1 not in R:
        return None
    r = R[year - 1]; pids = sorted(set(r.off) | set(r["def"]))
    for f, (num, wc) in FEATS.items():
        o, d, mu, h = ridge(r, pids, num, wc, LAMS[f])
        om = dict(zip(pids, o)); dm = dict(zip(pids, d))
        lo, hi = np.percentile(o, 20), np.percentile(d, 80)  # newcomers start weak
        out[f] = np.r_[[om.get(t, lo) for t in ids], [dm.get(t, hi) for t in ids]] * shrink
    return out


def walk(year, shrink=0.9, halflife=None):
    rows = R[year]; games = D[year][1]; ids = sorted(D[year][2].team_id)
    pri = final_prior(year, ids, shrink)
    games = games.assign(key=games.season_type * 100 + games.week)
    out = []
    for key, gw in games.groupby("key"):
        start = gw.date.min()
        hist = rows[rows.date < start]
        dw = None
        if halflife and len(hist):
            age = (pd.Timestamp(start) - hist.dt).dt.days.to_numpy()
            dw = 0.5 ** (age / halflife)
        nets = {}
        for f, (num, wc) in FEATS.items():
            p = None if pri is None else pri[f]
            if len(hist) == 0 and p is None:
                break
            o, d, mu, h = ridge(hist, ids, num, wc, LAMS[f], p, dw) if len(hist) else (p[:len(ids)], p[len(ids):], 0, 0)
            nets[f] = dict(zip(ids, o - d))
        if len(nets) < len(FEATS):
            continue
        for g in gw.itertuples():
            rec = {"year": year, "key": key, "gid": g.game_id, "home": 0 if g.neutral_site else 1,
                   "margin": g.home_score - g.away_score, "vegas": g.home_spread_margin, "ngames": len(hist)}
            for f in FEATS:
                rec[f] = nets[f][g.home_id] - nets[f][g.away_id]
            out.append(rec)
    return pd.DataFrame(out)
