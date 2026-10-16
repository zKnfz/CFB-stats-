"""Quarterback context from the play by play passer names.

`python3 qb.py` (in CI, after export3.py): for every team this season, who threw most of its passes last
game (and what share), and who leads the team in attempts so far. Writes site['qb'] in site_data.json:
{team_id: [last game's main passer, his share of that game's passes, season attempts leader, leader's attempts,
last game's date]}. Who starts tonight isn't in the data; this only shows whether the position changed.
`python3 qb.py --all` also builds data/qb.pkl (main passer for every game since 2004) for research.

Tested (Oct 2026, 13,306 lined games 2008 on): games right after a team's main passer changed were NOT
worse for the model against the line (model minus line average miss +0.20 points vs +0.28 in games without
a change; 5+ point model edges covered 53.1% vs 53.7%). So the page shows this as context, not as a warning
that the model is less reliable.
"""
import json
import sys; sys.path.insert(0, '.')
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from paths import DATA_DIR
from season import SEASON


def season(y):
    p = pd.read_parquet(f"{DATA_DIR}/pbp_{y}.parquet", columns=["game_id", "pos_team_id", "passer_player_name", "pass"])
    p = p[(p["pass"] == True) & p.passer_player_name.notna() & p.pos_team_id.notna()]
    c = p.groupby(["game_id", "pos_team_id", "passer_player_name"]).size().rename("att").reset_index()
    tot = c.groupby(["game_id", "pos_team_id"]).att.transform("sum")
    c["share"] = c.att / tot
    top = c.sort_values("att", ascending=False).drop_duplicates(["game_id", "pos_team_id"])
    top = top.rename(columns={"pos_team_id": "team", "passer_player_name": "qb"})
    top["team"] = top.team.astype(int); top["game_id"] = top.game_id.astype(int); top["year"] = y
    allp = c.rename(columns={"pos_team_id": "team", "passer_player_name": "qb"}); allp["team"] = allp.team.astype(int); allp["game_id"] = allp.game_id.astype(int); allp["year"] = y
    return top[["year", "game_id", "team", "qb", "att", "share"]], allp[["year", "game_id", "team", "qb", "att"]]


def site_qb(y):
    top, allp = season(y)
    sc = pd.read_parquet(f"{DATA_DIR}/sched_{y}.parquet", columns=["game_id", "game_date"])
    sc["game_id"] = sc.game_id.astype(int)
    sc["d"] = pd.to_datetime(sc.game_date, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    top = top.merge(sc[["game_id", "d"]], on="game_id").sort_values("d")
    out = {}
    for t, gs in top.groupby("team"):
        last = gs.iloc[-1]
        lead = allp[allp.team == t].groupby("qb").att.sum().sort_values(ascending=False)
        out[str(int(t))] = [last.qb, round(float(last.share), 2), lead.index[0], int(lead.iloc[0]), last.d]
    return out


if __name__ == "__main__":
    if "--all" in sys.argv:
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(season, range(2004, SEASON + 1)))
        top = pd.concat([r[0] for r in res]); allp = pd.concat([r[1] for r in res])
        pd.to_pickle({"top": top, "all": allp}, DATA_DIR + "/qb.pkl")
        print("qb rows", len(top), "games", top.game_id.nunique())
    site = json.load(open(DATA_DIR + "/site_data.json"))
    site["qb"] = site_qb(SEASON)
    print("qb context for", len(site["qb"]), "teams")
    json.dump(site, open(DATA_DIR + "/site_data.json", "w"), separators=(",", ":"))
