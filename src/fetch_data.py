"""Download play-by-play, schedules and team info for 2003 to CURRENT into ../data."""
import os, sys, urllib.request
from paths import DATA_DIR
CURRENT = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
REL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
RAW = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
os.makedirs(DATA_DIR, exist_ok=True)
def get(url, name, refresh=False):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path) and not refresh: return
    print("downloading", name); urllib.request.urlretrieve(url, path)
for y in range(2003, CURRENT + 1):
    cur = (y == CURRENT)  # always refresh the current season
    get(f"{RAW}/team_info/parquet/cfb_team_info_{y}.parquet", f"teaminfo_{y}.parquet", cur)
    if y >= 2004:
        get(f"{REL}/espn_cfb_pbp/play_by_play_{y}.parquet", f"pbp_{y}.parquet", cur)
        get(f"{REL}/espn_cfb_schedules/cfb_schedule_{y}.parquet", f"sched_{y}.parquet", cur)
# full current-season schedule including games not played yet
get(f"{RAW}/schedules/parquet/cfb_schedules_{CURRENT}.parquet", f"up_cfb_schedules_{CURRENT}.parquet", True)
