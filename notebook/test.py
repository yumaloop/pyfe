import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader

from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from models import MarkowitzMinVarianceModel


df_tpx500 = pd.read_csv("../data/master/universe_tse_topix500_20011031-20201031.csv")
s = df_tpx500["20111031"].values
s = s[~np.isnan(s)] # delete nan
symbols_tpx500 = s.astype(np.int64).astype(str) # cast as str type
symbols_tpx500 = [str(s)+'.T' for s in symbols_tpx500]

st = '2011/10/31' 
ed = '2020/10/31' 

st = datetime(2011, 10, 31)
ed = datetime(2020, 10, 31)

dfs = []
for symbol in tqdm(symbols_tpx500):
    try:
        df = pandas_datareader.data.DataReader(symbol, 'yahoo', st, ed) # daily
        df = df.sort_values(by='Date', ascending=True)
        df = df.resample('M').mean() # daily -> monthly
        df = df.fillna(method='ffill') # 1つ前の行の値で埋める
        df = df[['Close']].rename(columns={'Close': symbol})
        dfs.append(df)
    except:
        pass
    
df = pd.concat(dfs, axis=1)
df.to_csv("../data/tse_tpx500_20111031-202010131.csv", index=False)