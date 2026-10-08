import sys, itertools, pickle; sys.path.insert(0,'.')
from paths import DATA_DIR
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from model import walk
YEARS=list(range(2008,2027))
SHRINKS=(0.9,)
HALFLIVES=(None,)
def job(a): y,sh,hl=a; df=walk(y,sh,hl); df['shrink']=sh; df['hl']=hl or 0; return df
if __name__=='__main__':
    cfgs=[(y,sh,hl) for sh in SHRINKS for hl in HALFLIVES for y in YEARS]
    # model selection grid; the shipped model only needs shrink 0.9, no decay

    with ProcessPoolExecutor(8) as ex: out=pd.concat(ex.map(job,cfgs))
    out.to_pickle(DATA_DIR + '/bt.pkl'); print(len(out))
