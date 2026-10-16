"""Walk-forward feature generation for model selection (vectorized ridge)."""
from paths import DATA_DIR
import pickle, json
import numpy as np, pandas as pd

D = pickle.load(open(DATA_DIR + "/prep.pkl", "rb"))
try:
    PRESEASON = json.load(open(DATA_DIR + "/preseason.json"))
except FileNotFoundError:
    PRESEASON = {}

# talent/returning production z-score weight on the preseason prior, in std-devs of that
# feature's own net rating. Chosen by walk-forward backtest: tuned on 2015-2024 preseason-only
# games, confirmed on 2025-2026 (never touched while tuning). Cuts preseason-only MAE 15.44 -> 14.07
# and nearly erases the ~5 point too-generous-to-the-favorite bias. returning_ppa made things worse,
# alone or blended in, so it stays off. export2.py/export3.py pick this up since they share final_prior().
PRESEASON_COEF = {"talent": 0.8, "returning": 0.0}
FEATS = {  # name: (numerator col, weight col)
    "E": ("epa", "plays"), "NT": ("epa_nt", "plays_nt"), "SR": ("succ", "plays"), "PTS": ("pts", "one"),
    "EX": ("ex_nt", "plays_ex"),  # explosiveness: EPA per successful, turnover-free play
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


LAMS = {"E": 100, "NT": 100, "SR": 100, "PTS": 3, "EX": 150}
# display and matchup fingerprint stats, not in the rating. Same definitions and pulls as the page.
FP_FEATS = {"PE": ("pepa", "pplays"), "RE": ("repa", "rplays"), "HV": ("hav", "plays"), "FD": ("tpts", "trips")}
FP_LAMS = {"PE": 100, "RE": 100, "HV": 100, "FD": 10}
SNAP_FEATS = ["E", "NT", "SR", "PTS", "EX", "PE", "RE", "HV", "FD"]  # per team o and d saved for every game


def _preseason_adj(year, ids):
    """Per-team z-score blend of talent + returning production for the upcoming season,
    in raw (unscaled) units. None/0 for any team missing either stat, so it's a no-op
    unless PRESEASON_COEF is set. Scaled to each feature's own std in final_prior()."""
    yr = PRESEASON.get(str(year))
    if not yr:
        return None
    def zscore(key):
        vals = {int(t): v[key] for t, v in yr.items() if v.get(key) is not None}
        if len(vals) < 10:
            return {}
        arr = np.array(list(vals.values())); mu, sd = arr.mean(), arr.std()
        return {t: (x - mu) / sd for t, x in vals.items()} if sd > 0 else {}
    tz, rz = zscore("talent"), zscore("returning_ppa")
    if not tz and not rz:
        return None
    c = PRESEASON_COEF
    return {t: c["talent"] * tz.get(t, 0.0) + c["returning"] * rz.get(t, 0.0) for t in ids}


def final_prior(year, ids, shrink, feats=None):
    """Last season's final o,d per feature, shrunk; missing teams get a below-average default.
    Optionally nudged by this season's talent/returning production z-scores (PRESEASON_COEF).
    feats=FP_FEATS gives the fingerprint-only stats a prior too (no talent nudge: havoc allowed is
    a lower-is-better offense stat, so the nudge's sign would not mean the same thing)."""
    out = {}
    if year - 1 not in R:
        return None
    r = R[year - 1]; pids = sorted(set(r.off) | set(r["def"]))
    padj = _preseason_adj(year, ids) if feats is None else None
    lams = LAMS if feats is None else FP_LAMS
    for f, (num, wc) in (FEATS if feats is None else feats).items():
        o, d, mu, h = ridge(r, pids, num, wc, lams[f])
        om = dict(zip(pids, o)); dm = dict(zip(pids, d))
        lo, hi = np.percentile(o, 20), np.percentile(d, 80)  # newcomers start weak
        base_o = np.array([om.get(t, lo) for t in ids])
        base_d = np.array([dm.get(t, hi) for t in ids])
        if padj:
            std_f = (base_o - base_d).std()
            adj = np.array([padj.get(t, 0.0) for t in ids]) * std_f
            base_o = base_o + 0.5 * adj
            base_d = base_d - 0.5 * adj
        out[f] = np.r_[base_o, base_d] * shrink
    return out


def snapshot(hist, ids, pri, pri_fp, gp_rows=None):
    """Every team's o and d for SNAP_FEATS from `hist` only (deviations from that fit's league
    average, so eras compare), plus games played and opponents' average NT/SR/PTS nets so far
    (SOS, before beta). pri/pri_fp: priors from final_prior (None = shrink to average).
    Returns {team: {"E": (o, d), ..., "gp": n, "sos": (NT, SR, PTS) or None}}."""
    n = len(ids); vals = {}
    for f in SNAP_FEATS:
        num, wc = FEATS[f] if f in FEATS else FP_FEATS[f]
        lam = LAMS[f] if f in LAMS else FP_LAMS[f]
        p = (pri if f in FEATS else pri_fp)
        p = None if p is None else p[f]
        if len(hist):
            o, d, mu, h = ridge(hist, ids, num, wc, lam, p)
        else:
            o, d = (p[:n], p[n:]) if p is not None else (np.zeros(n), np.zeros(n))
        vals[f] = (o, d)
    idx = {t: i for i, t in enumerate(ids)}
    net = {f: vals[f][0] - vals[f][1] for f in ("NT", "SR", "PTS")}
    opps = {}
    for off, de in zip(hist.off, hist["def"]):  # one row per team per game, so each game counts once per side
        opps.setdefault(off, []).append(de)
    out = {}
    for t, i in idx.items():
        op = [idx[x] for x in opps.get(t, []) if x in idx]
        out[t] = {f: (float(vals[f][0][i]), float(vals[f][1][i])) for f in SNAP_FEATS}
        out[t]["gp"] = len(opps.get(t, []))
        out[t]["sos"] = tuple(float(net[f][op].mean()) for f in ("NT", "SR", "PTS")) if op else None
    return out


def walk(year, shrink=0.9, halflife=None, snap=False):
    """Pregame features for every game: only games from before that week are used.
    snap=True also stores each side's full pregame team snapshot (see snapshot()) as h_*/a_* columns,
    for the matchup fingerprints in export3.py."""
    rows = R[year]; games = D[year][1]; ids = sorted(D[year][2].team_id)
    pri = final_prior(year, ids, shrink)
    pri_fp = final_prior(year, ids, shrink, FP_FEATS) if snap else None
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
            if f == "PTS":  # pregame projected total: both teams' expected points; home field cancels out
                if not len(hist):
                    mu = R[year - 1].pts.mean() if year - 1 in R else 28.0
                po, pd_, mu_pts = dict(zip(ids, o)), dict(zip(ids, d)), mu
        if len(nets) < len(FEATS):
            continue
        sn = snapshot(hist, ids, pri, pri_fp) if snap else None
        for g in gw.itertuples():
            rec = {"year": year, "key": key, "gid": g.game_id, "home": 0 if g.neutral_site else 1,
                   "margin": g.home_score - g.away_score, "vegas": g.home_spread_margin, "ngames": len(hist)}
            for f in FEATS:
                rec[f] = nets[f][g.home_id] - nets[f][g.away_id]
            rec["ptot"] = 2 * mu_pts + po[g.home_id] + pd_[g.away_id] + po[g.away_id] + pd_[g.home_id]
            if sn:
                for side, t in (("h", g.home_id), ("a", g.away_id)):
                    s_ = sn[t]
                    for f in SNAP_FEATS:
                        rec[f"{side}_{f}_o"], rec[f"{side}_{f}_d"] = s_[f]
                    rec[f"{side}_gp"] = s_["gp"]
                    for f, v in zip(("NT", "SR", "PTS"), s_["sos"] or (np.nan,) * 3):
                        rec[f"{side}_sos_{f}"] = v
            out.append(rec)
    return pd.DataFrame(out)
