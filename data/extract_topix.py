import pandas as pd

# all stocks (in TSE)
df = pd.read_csv("./tse_allstocks_20201030.csv")

# extract all stocks of TOPIX Core30
df_topix_core30 = df[df["規模区分"] == "TOPIX Core30"]
df_topix_core30.to_csv("./tse_topixcore30_20201030.csv", index=False)

# extract all stocks of TOPIX 500
topix500_frags = ["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"]
df_topix500 = df[df["規模区分"].str.contains('|'.join(frag for frag in topix500_frags))]
df_topix500.to_csv("./tse_topix500_20201030.csv", index=False)
