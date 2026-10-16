"""Player game logs from the play by play, for the Outlier style player props on game pages.

For this season and last: every passer, rusher and receiver's line in every game (FBS and FCS opponents):
pass attempts, completions, passing yards, passing TDs, interceptions, rushes, rushing yards, rushing TDs,
receptions, receiving yards, receiving TDs. Sacks are not counted as rushes (college box scores count them
as rushes; Outlier style props follow the sportsbook convention, which doesn't). Players are kept if they
had real volume in either season, so the page stays small.

Writes into site_data.json:
  'players': {pid: [name, team id of his latest game]}
  'plog': [[pid, game_id, date, team, opp, home(1/0/-1 neutral as 0), att, cmp, pyd, ptd, int, ra, ryd, rtd, rec, wyd, wtd, targets], ...]
There are no sportsbook prop lines in the data; the page lets you type a line and grades history against it.
Run after export3.py.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from paths import DATA_DIR
from season import SEASON

COLS = ["game_id", "pos_team_id", "def_pos_team_id", "type.text", "statYardage", "completion", "pass_td", "rush_td",
        "pass_attempt", "rush", "sack", "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
        "receiver_player_id", "receiver_player_name"]
NULLIFIED = "Penalty|Timeout|Kickoff|Punt|Field Goal|Extra Point|Two Point|End of|Coin Toss"


def season(y):
    p = pd.read_parquet(f"{DATA_DIR}/pbp_{y}.parquet", columns=COLS)
    p = p[~p["type.text"].str.contains(NULLIFIED, na=False) & p.pos_team_id.notna()].copy()
    p["yds"] = pd.to_numeric(p.statYardage, errors="coerce").fillna(0)
    p["int"] = p["type.text"].str.contains("Interception", na=False)
    is_sack = (p.sack == True) | p["type.text"].str.contains("Sack", na=False)
    out = []
    # passing: attempts are passes with a passer that weren't sacks
    ps = p[(p.pass_attempt == True) & ~is_sack & p.passer_player_id.notna()]
    out.append(ps.assign(pid=ps.passer_player_id, name=ps.passer_player_name, att=1, cmp=(ps.completion == True).astype(int),
                         pyd=np.where(ps.completion == True, ps.yds, 0), ptd=(ps.pass_td == True).astype(int), int_=ps["int"].astype(int)))
    # rushing: designed runs and scrambles, not sacks
    rs = p[(p.rush == True) & ~is_sack & p.rusher_player_id.notna()]
    out.append(rs.assign(pid=rs.rusher_player_id, name=rs.rusher_player_name, ra=1, ryd=rs.yds, rtd=(rs.rush_td == True).astype(int)))
    # receiving: every pass thrown to a named receiver is a target, so games where he was targeted but shut out
    # still show up as zeros (games with no target at all can't be seen in play by play)
    wr = p[(p.pass_attempt == True) & ~is_sack & p.receiver_player_id.notna()]
    cm = wr.completion == True
    out.append(wr.assign(pid=wr.receiver_player_id, name=wr.receiver_player_name, tgt=1, rec=cm.astype(int),
                         wyd=np.where(cm, wr.yds, 0), wtd=(cm & (wr.pass_td == True)).astype(int)))
    stats = ["att", "cmp", "pyd", "ptd", "int_", "ra", "ryd", "rtd", "rec", "wyd", "wtd", "tgt"]
    d = pd.concat(out, ignore_index=True)
    d[stats] = d.reindex(columns=stats).fillna(0)
    d["pid"] = d.pid.astype(str).str.replace(r"\.0$", "", regex=True)
    g = d.groupby(["pid", "game_id", "pos_team_id", "def_pos_team_id"]).agg(name=("name", "first"), **{s: (s, "sum") for s in stats}).reset_index()
    g["year"] = y
    return g


def main():
    sc = pd.concat([pd.read_parquet(f"{DATA_DIR}/sched_{y}.parquet", columns=["game_id", "game_date", "home_id", "neutral_site"]) for y in (SEASON - 1, SEASON)])
    sc["game_id"] = sc.game_id.astype(int); sc["home_id"] = sc.home_id.astype(int)
    sc["d"] = pd.to_datetime(sc.game_date, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    sc = sc.drop_duplicates("game_id").set_index("game_id")
    g = pd.concat([season(SEASON - 1), season(SEASON)], ignore_index=True)
    g["game_id"] = g.game_id.astype(int); g = g[g.game_id.isin(sc.index)]
    # volume filter: a QB, runner or receiver in either season
    tot = g.groupby(["pid", "year"])[["att", "ra", "rec"]].sum()
    keep = set(tot[(tot.att >= 40) | (tot.ra >= 25) | (tot.rec >= 10)].reset_index().pid)
    g = g[g.pid.isin(keep)]
    g["d"] = g.game_id.map(sc.d)
    g = g.sort_values(["d", "game_id"])
    site = json.load(open(DATA_DIR + "/site_data.json"))
    latest = g.groupby("pid").last()
    site["players"] = {pid: [r["name"], int(r.pos_team_id)] for pid, r in latest.iterrows()}
    rows = []
    for r in g.itertuples():
        team, opp = int(r.pos_team_id), int(r.def_pos_team_id)
        neu, hid = bool(sc.at[r.game_id, "neutral_site"]), int(sc.at[r.game_id, "home_id"])
        rows.append([r.pid, int(r.game_id), r.d, team, opp, 0 if neu else (1 if hid == team else -1),
                     int(r.att), int(r.cmp), int(r.pyd), int(r.ptd), int(r.int_), int(r.ra), int(r.ryd), int(r.rtd), int(r.rec), int(r.wyd), int(r.wtd), int(r.tgt)])
    site["plog"] = rows
    print(f"players: {len(site['players'])} players, {len(rows)} player games ({SEASON - 1} and {SEASON})")
    json.dump(site, open(DATA_DIR + "/site_data.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
