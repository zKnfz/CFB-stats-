"""Walk-forward pregame features for 2004-2007 (bt.pkl from run_bt.py covers 2008+) -> data/pregame.pkl"""
import sys; sys.path.insert(0, '.')
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from paths import DATA_DIR
from model import walk
if __name__ == "__main__":
    with ProcessPoolExecutor() as ex: old = pd.concat(ex.map(walk, [2004, 2005, 2006, 2007], [0.9] * 4))
    old["shrink"], old["hl"] = 0.9, 0
    bt = pd.read_pickle(DATA_DIR + "/bt.pkl"); bt = bt[(bt.shrink == 0.9) & (bt.hl == 0)]
    pd.concat([old, bt]).to_pickle(DATA_DIR + "/pregame.pkl"); print("pregame rows saved")
