"""CFB T-Rank style ratings: data prep + opponent-adjusted ridge ratings."""
from paths import DATA_DIR
import numpy as np
import pandas as pd

DATA = DATA_DIR
PBP_COLS = ["game_id", "pos_team_id", "def_pos_team_id", "period", "scrimmage_play",
            "kneel_down", "EPA", "EPA_success", "rush", "pass", "type.text", "turnover_vec",
            "pos_score_diff_start", "gameSpread", "homeFavorite"]
# Garbage time (Connelly-style): score margin at snap above these, by quarter
GARBAGE = {1: 999, 2: 38, 3: 28, 4: 22}


def load_season(year):
    info = pd.read_parquet(f"{DATA}/teaminfo_{year}.parquet")
    fbs = info[info.classification == "fbs"]
    fbs_ids = set(fbs.team_id.astype(int))
    sched = pd.read_parquet(f"{DATA}/sched_{year}.parquet")
    sched = sched[sched.status == "STATUS_FINAL"].copy()
    sched["home_id"] = sched.home_id.astype(int)
    sched["away_id"] = sched.away_id.astype(int)
    sched = sched[sched.home_id.isin(fbs_ids) & sched.away_id.isin(fbs_ids)]
    sched["date"] = pd.to_datetime(sched.game_date).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")

    p = pd.read_parquet(f"{DATA}/pbp_{year}.parquet", columns=PBP_COLS)
    p = p[p.game_id.isin(sched.game_id)]
    spread = p.groupby("game_id").agg(gs=("gameSpread", "first"), hf=("homeFavorite", "first"))
    spread["home_spread_margin"] = np.where(spread.hf == True, spread.gs, -spread.gs)
    spread.loc[spread.gs.isna(), "home_spread_margin"] = np.nan

    ok = ((p.scrimmage_play == True) & ((p.rush == True) | (p["pass"] == True)) & (p.kneel_down != True)
          & ~p["type.text"].str.contains("Two Point|2pt|Punt", na=False) & p.EPA.notna())
    gt = p.period.map(GARBAGE).fillna(999)
    ok &= p.pos_score_diff_start.abs() <= gt
    p = p[ok].copy()
    p["succ"] = (p.EPA_success == True).astype(float)
    p["is_pass"] = (p["pass"] == True)
    p["tov"] = (p.turnover_vec.astype(float) > 0)
    p["epa_nt"] = np.where(p.tov, np.nan, p.EPA)
    p["pos_team_id"] = p.pos_team_id.astype(int)
    p["def_pos_team_id"] = p.def_pos_team_id.astype(int)

    agg = p.groupby(["game_id", "pos_team_id", "def_pos_team_id"]).agg(
        plays=("EPA", "size"), epa=("EPA", "sum"), succ=("succ", "sum"),
        pplays=("is_pass", "sum"), tov=("tov", "sum"), epa_nt=("epa_nt", "sum"), plays_nt=("epa_nt", "count"),
        pepa=("EPA", lambda s: s[p.loc[s.index, "is_pass"]].sum())).reset_index()
    agg["rplays"] = agg.plays - agg.pplays
    agg["repa"] = agg.epa - agg.pepa
    agg = agg.merge(sched[["game_id", "date", "week", "season_type", "neutral_site", "home_id", "away_id",
                           "home_score", "away_score"]], on="game_id")
    agg["loc"] = np.where(agg.neutral_site, 0, np.where(agg.pos_team_id == agg.home_id, 1, -1))
    agg = agg.rename(columns={"pos_team_id": "off", "def_pos_team_id": "def"})

    games = sched[["game_id", "date", "week", "season_type", "neutral_site", "home_id", "away_id",
                   "home_score", "away_score"]].merge(spread[["home_spread_margin"]], left_on="game_id",
                                                       right_index=True, how="left")
    # keep only games that have both offensive sides
    both = agg.groupby("game_id").size()
    games = games[games.game_id.isin(both[both == 2].index)]
    agg = agg[agg.game_id.isin(games.game_id)]
    teams = fbs[["team_id", "school", "abbreviation", "conference", "color", "alt_color"]].copy()
    teams["team_id"] = teams.team_id.astype(int)
    return agg.reset_index(drop=True), games.reset_index(drop=True), teams.reset_index(drop=True)


def fit(rows, team_ids, col="epa", lam=200.0, prior_o=None, prior_d=None, hfa=True):
    """Weighted ridge: rate = mu + o[off] + d[def] + h*loc, weights = plays.
    Penalty pulls o,d toward priors (default 0). Returns dict of arrays."""
    idx = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)
    r = rows[rows.off.isin(idx) & rows["def"].isin(idx)]
    k = 2 * n + 2  # o..., d..., mu, h
    A = np.zeros((k, k)); b = np.zeros(k)
    po = np.zeros(n) if prior_o is None else prior_o
    pd_ = np.zeros(n) if prior_d is None else prior_d
    for off, de, loc, w, s in zip(r.off.map(idx), r["def"].map(idx), r["loc"], r.plays, r[col]):
        y = s / w
        cols = [off, n + de, 2 * n]
        vals = [1.0, 1.0, 1.0]
        if hfa and loc != 0:
            cols.append(2 * n + 1); vals.append(float(loc))
        for c1, v1 in zip(cols, vals):
            b[c1] += w * v1 * y
            for c2, v2 in zip(cols, vals):
                A[c1, c2] += w * v1 * v2
    for i in range(n):
        A[i, i] += lam; b[i] += lam * po[i]
        A[n + i, n + i] += lam; b[n + i] += lam * pd_[i]
    A[2 * n, 2 * n] += 1e-6
    A[2 * n + 1, 2 * n + 1] += 1e-6 if hfa else 1e9
    x = np.linalg.solve(A, b)
    mu = x[2 * n]
    return {"o": x[:n], "d": x[n:2 * n], "mu": mu, "h": x[2 * n + 1],
            "adjO": mu + x[:n], "adjD": mu + x[n:2 * n]}
