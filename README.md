# afe2_backtest

py backtest repo

### pypi packages

```
pip3 install yfinance
pip3 install pandas_datareader
pip3 install cvxopt
```

### sample

```
import datetime
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

with open('data/temp/alpha_vantage_api_key.txt') as f:
    api_key = f.read()

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2019, 12, 31)

df_sne = web.DataReader('SNE', 'av-daily', start, end, api_key=api_key)
print(df_sne)
#              open    high    low  close   volume
# 2015-01-02  20.47  20.685  20.43  20.56  1229939
# 2015-01-05  20.45  20.450  20.21  20.26  1083137
# 2015-01-06  20.46  20.580  20.15  20.25  2209124
# 2015-01-07  21.59  21.700  21.47  21.53  2486293
# 2015-01-08  21.53  21.620  21.47  21.56  1296471
# ...           ...     ...    ...    ...      ...
# 2019-12-24  67.98  68.000  67.76  67.76   264463
# 2019-12-26  68.00  68.030  67.85  68.02   517975
# 2019-12-27  68.03  68.100  67.73  67.78   351118
# 2019-12-30  67.78  67.790  67.25  67.72   993865
# 2019-12-31  67.72  68.025  67.51  68.00   549672
#
# [1258 rows x 5 columns]
```

# Data resources

### Index

名称    値  前日比  前日比(%)   1ヶ月利回り変化幅   年間利回り変化幅    更新日時 (JST)
CTRN:IND
ﾅｽﾀﾞｯｸ 運輸株指数
5,411.06    +100.63 +1.89%  +5.47%  +2.39%  7:16
SML:IND
S&P 小型株600種
994.29  +25.79  +2.66%  +7.64%  +1.18%  6:15
RTY:IND
ﾗｯｾﾙ2000種指数
1,744.04    +35.57  +2.08%  +6.75%  +9.24%  9:00
NBI:IND
ﾅｽﾀﾞｯｸ ﾊﾞｲｵﾃｸﾉﾛｼﾞｰ株指数
4,436.25    +52.62  +1.20%  +0.99%  +26.65% 7:16
INDU:IND
NYﾀﾞｳ 工業株30種
29,479.81   +399.64 +1.37%  +3.05%  +5.27%  6:59
CBNK:IND
ﾅｽﾀﾞｯｸ 銀行株指数
3,239.77    +84.23  +2.67%  +17.65% -15.23% 7:16
BBREIT:IND
ﾌﾞﾙｰﾑﾊﾞｰｸﾞ REIT指数
293.47  +7.28   +2.54%  +5.33%  -8.26%  6:00
NYA:IND
NYSE 総合指数
13,761.32   +209.86 +1.55%  +4.50%  +1.99%  7:59
NDX:IND
ﾅｽﾀﾞｯｸ 100指数
11,937.84   +110.70 +0.94%  +0.72%  +43.56% 7:16
CFIN:IND
ﾅｽﾀﾞｯｸ 金融株指数
10,025.96   +152.27 +1.54%  +0.74%  +9.56%  7:16
CINS:IND
ﾅｽﾀﾞｯｸ 保険株指数
9,517.48    +105.04 +1.12%  +2.37%  -3.90%  7:16
TRAN:IND
NYﾀﾞｳ 輸送株20種
12,085.32   +246.57 +2.08%  +2.10%  +11.12% 6:59
CUTL:IND
ﾅｽﾀﾞｯｸ 通信株指数
441.01  +11.34  +2.64%  +5.78%  +18.15% 7:16
SPX:IND
S&P 500種
3,585.15    +48.14  +1.36%  +2.91%  +14.89% 6:59
CCMP:IND
ﾅｽﾀﾞｯｸ 総合指数
11,829.29   +119.70 +1.02%  +1.35%  +38.50% 7:16
UTIL:IND
NYﾀﾞｳ 公共株15種
910.91  +8.81   +0.98%  +2.98%  +7.36%  6:59
BKX:IND
KBW銀行株指数
87.53   +1.89   +2.21%  +12.12% -19.30% 7:16
RIY:IND
ﾗｯｾﾙ1000種指数
2,000.69    +25.97  +1.32%  +2.76%  +16.06% 9:00
NDF:IND
ﾅｽﾀﾞｯｸ 金融100指数
4,857.14    +86.09  +1.80%  +5.47%  -2.39%  7:16
RAY:IND
ﾗｯｾﾙ3000種指数
2,112.38    +28.36  +1.36%  +2.99%  +15.61% 9:00
IXK:IND
ﾅｽﾀﾞｯｸ ｺﾝﾋﾟｭｰﾀｰ株指数
8,229.65    +50.55  +0.62%  +1.17%  +47.73% 7:16
MID:IND
S&P 中型株400種
2,113.26    +43.04  +2.08%  +5.80%  +5.63%  8:59
CIND:IND
ﾅｽﾀﾞｯｸ 工業株指数
9,194.40    +105.19 +1.16%  -0.08%  +40.72% 7:16




Dow Jones Industrial Average (DJIA)

- 
- Japan
    - TOPIX500
    - NIKKEI225
- United States
    - S&P 500 Index (SP500)
    - Dow Jones Industrial Average (DJIA)
    - ...

### Market data

- Japan
    - TSE, (Tokyo Stock Exchange)<br>
        - Operating company: Japan Exchange Group, Inc. (JPX)<br>
          https://www.jpx.co.jp/
        - 統計情報(株式関連)<br>
          https://www.jpx.co.jp/markets/statistics-equities/index.html
        - Stock list: 東証上場銘柄一覧<br>
          https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
- United States
    - NYSE (New York Stock Exchange)<br>
        - Operating company: Intercontinental Exchange (ICE)<br>
          https://www.theice.com/index
        - Stock list<br>
          https://www.nyse.com/listings_directory/stock
    - NASDAQ (National Association of Securities Dealers Automated Quotations)
        - Operating company: Nasdaq, Inc.
          https://www.nasdaq.com/
        - Stock list<br>
          ...


