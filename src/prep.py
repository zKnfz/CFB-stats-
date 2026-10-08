"""Turn raw play-by-play into per team-game rows for every season -> data/prep.pkl"""
import sys, pickle, os; sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor
from paths import DATA_DIR
from build import load_season
if __name__ == "__main__":
    ys = sorted(int(f[4:8]) for f in os.listdir(DATA_DIR) if f.startswith("pbp_"))
    with ProcessPoolExecutor() as ex: res = list(ex.map(load_season, ys))
    pickle.dump(dict(zip(ys, res)), open(os.path.join(DATA_DIR, "prep.pkl"), "wb")); print("prepped", ys[0], "to", ys[-1])
