import csv
import pandas as pd
import pandas_datareader

def get_nasdaq_symbols():
    df_nasdaq = pandas_datareader.nasdaq_trader.get_nasdaq_symbols()
    df_nasdaq['Security Name'] = df_nasdaq['Security Name'].astype(str)
    df_nasdaq.to_csv("./nasdaq_allstocks_2020.csv", index=False)

def get_topix_symbols():
    # all stocks (in TSE)
    df = pd.read_csv("./tse_allstocks_20201030.csv")

    # extract all stocks of TOPIX Core30
    df_topix_core30 = df[df["規模区分"] == "TOPIX Core30"]
    df_topix_core30.to_csv("./tse_topixcore30_20201030.csv", index=False)

    # extract all stocks of TOPIX 500
    topix500_frags = ["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"]
    df_topix500 = df[df["規模区分"].str.contains('|'.join(frag for frag in topix500_frags))]
    df_topix500.to_csv("./tse_topix500_20201030.csv", index=False)
