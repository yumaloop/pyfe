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

# all symbols in TOPIX 500 on 2011/10/31
df_tpx500 = pd.read_csv("../data/master/universe_tse_topix500_20011031-20201031.csv")
s = df_tpx500["20111031"].values
s = s[~np.isnan(s)] # delete nan
symbols_tpx500 = s.astype(np.int64).astype(str) # cast as str type
symbols_tpx500 = [str(s)+'.T' for s in symbols_tpx500]


# df: time-series price data of stocks in `symbols_tpx500`
st = datetime(2000, 10, 31)
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

# fill nan
for col in df.columns:
    st_idx = df[col].first_valid_index()
    ed_idx = df[col].last_valid_index()

    # for any columns (=stocks)
    if df[col].isnull().any():
        # 新規上場
        if st_idx != df.index[0]:
            df[col] = df[col].fillna(df[col][st_idx])

        # 上場廃止
        if df.index[-1] != ed_idx:
            df[col] = df[col].fillna(df[col][ed_idx])

# save df as .csv file
df.to_csv("./tse_topix500_20001031-202010131.csv", index=True)
