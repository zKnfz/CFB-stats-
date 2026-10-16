"""Pull preseason signal from CollegeFootballData: team talent composite and
returning production. Needs a free API key in the CFBD_API_KEY env var
(https://collegefootballdata.com/key). Writes data/preseason.json:
{year: {team_id: {"talent": float|null, "returning_ppa": float|null}}}
for the seasons requested, matched to our own team id/name list.
Only fetches seasons missing from the committed preseason.json, since talent and
returning production don't change once a season starts and CFBD's free tier caps calls.
"""
import sys, os, json, unicodedata, urllib.request
sys.path.insert(0, '.')
from paths import DATA_DIR
from season import SEASON

API_KEY = os.environ.get("CFBD_API_KEY")
BASE = "https://api.collegefootballdata.com"

ALIASES = {
    "appalachian state": "app state", "southern mississippi": "southern miss",
    "hawaii": "hawai i", "louisiana monroe": "ul monroe", "louisiana lafayette": "louisiana",
    "connecticut": "uconn", "miami": "miami", "ole miss": "ole miss",
}

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", " ").replace("-", " ")
    s = " ".join(s.split())
    if s.endswith(" st"):
        s = s[:-3] + " state"
    return s

def cfbd_get(path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def main():
    path = DATA_DIR + '/preseason.json'
    try:
        out = json.load(open(path))
    except (FileNotFoundError, ValueError):
        out = {}
    years = [y for y in range(2015, SEASON + 1) if str(y) not in out]
    if not years:
        print(f"preseason.json already has 2015 to {SEASON}, no CFBD calls needed")
        return
    if not API_KEY:
        print("No CFBD_API_KEY set, skipping preseason fetch for", years)
        return

    site = json.load(open(DATA_DIR + '/site_data.json'))
    norm_to_id = {norm(v[0]): int(k) for k, v in site['teams'].items()}

    def match(name):
        n = ALIASES.get(norm(name), norm(name))
        return norm_to_id.get(n)

    talent_unmatched, ret_unmatched = set(), set()
    talent_n, ret_n = 0, 0

    for y in years:
        year_out = {}
        try:
            talent = cfbd_get("/talent", year=y)
            for row in talent:
                tid = match(row["team"])
                if tid is None:
                    talent_unmatched.add(row["team"]); continue
                year_out.setdefault(tid, {})["talent"] = row["talent"]
                talent_n += 1
        except Exception as e:
            print(f"talent fetch failed for {y}:", e)

        try:
            ret = cfbd_get("/player/returning", year=y)
            for row in ret:
                tid = match(row["team"])
                if tid is None:
                    ret_unmatched.add(row["team"]); continue
                year_out.setdefault(tid, {})["returning_ppa"] = row.get("totalPPA") or row.get("percentPPA")
                ret_n += 1
        except Exception as e:
            print(f"returning production fetch failed for {y}:", e)

        if year_out:
            out[str(y)] = year_out

    print(f"talent rows matched: {talent_n}, returning production rows matched: {ret_n}")
    print(f"unmatched talent team names ({len(talent_unmatched)}):", sorted(talent_unmatched)[:20])
    print(f"unmatched returning-production team names ({len(ret_unmatched)}):", sorted(ret_unmatched)[:20])
    json.dump(out, open(path, 'w'))

if __name__ == "__main__":
    main()
