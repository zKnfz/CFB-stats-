"""Look-ahead check for the Games Like This fingerprints. Run from src/: python3 test_analogs.py

1. Poison test: blow up one mid-season game's own stats and score, rerun that season's walk forward,
   and confirm that game's fingerprint (and every game that week or earlier) is unchanged, while
   later weeks do change. If a game could see its own result or the future, the first part fails.
2. Rebuild test: for a sample of games, rebuild the snapshot by hand from only the rows dated before
   that week started and confirm it matches what walk() stored.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import model
from model import walk, snapshot, final_prior, FP_FEATS, SNAP_FEATS, R, D

YEAR = 2019
snap_cols = lambda df: [c for c in df.columns if c[:2] in ("h_", "a_")]


def poison_test():
    base = walk(YEAR, snap=True).set_index("gid")
    games = D[YEAR][1].assign(key=lambda g: g.season_type * 100 + g.week)
    target = games[games.key == 208].iloc[0]  # a regular season week 8 game
    r = R[YEAR].copy(); m = r.game_id == target.game_id
    r.loc[m, ["epa", "epa_nt", "succ", "ex_nt", "pepa", "repa", "hav", "tpts"]] *= 25
    r.loc[m, "pts"] += 70
    orig = model.R[YEAR]; model.R[YEAR] = r
    try:
        bad = walk(YEAR, snap=True).set_index("gid")
    finally:
        model.R[YEAR] = orig
    cols = snap_cols(base); keys = games.set_index("game_id").key
    same = bad.loc[base.index, cols].sub(base[cols]).abs().max(axis=1) < 1e-9
    upto = base.index[keys.reindex(base.index) <= 208]
    later = base.index[keys.reindex(base.index) > 208]
    assert same[target.game_id], "the poisoned game's own fingerprint moved: it can see its own result"
    assert same[upto].all(), f"{(~same[upto]).sum()} games that week or earlier moved"
    assert (~same[later]).mean() > 0.5, "later games did not move, so the poison test is not testing anything"
    print(f"poison test ok: game {target.game_id} and {len(upto)} games through week 8 unchanged, "
          f"{(~same[later]).sum()} of {len(later)} later games changed")


def rebuild_test(n=25):
    base = walk(YEAR, snap=True).set_index("gid")
    games = D[YEAR][1].assign(key=lambda g: g.season_type * 100 + g.week)
    ids = sorted(D[YEAR][2].team_id); pri = final_prior(YEAR, ids, 0.9); pfp = final_prior(YEAR, ids, 0.9, FP_FEATS)
    rows = R[YEAR]; rng = np.random.default_rng(0)
    for gid in rng.choice(base.index, n, replace=False):
        g = games[games.game_id == gid].iloc[0]
        start = games[games.key == g.key].date.min()
        hist = rows[rows.date < start]
        assert (hist.date < g.date).all() and gid not in set(hist.game_id)
        s = snapshot(hist, ids, pri, pfp)
        for side, t in (("h", g.home_id), ("a", g.away_id)):
            for f in SNAP_FEATS:
                assert abs(s[t][f][0] - base.loc[gid, f"{side}_{f}_o"]) < 1e-9
            assert s[t]["gp"] == base.loc[gid, f"{side}_gp"]
    print(f"rebuild test ok: {n} games match a snapshot built only from games before their week")


if __name__ == "__main__":
    poison_test()
    rebuild_test()
