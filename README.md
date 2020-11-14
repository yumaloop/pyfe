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

http://www.eoddata.com/symbols.aspx?AspxAutoDetectCookieSupport=1

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
    - NASDAQ (National Association of Securities Dealers Automated Quotations)
        - Operating company: Nasdaq, Inc.
          https://www.nasdaq.com/
        - List of all stocks<br>
          download link: https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download
    - NYSE (New York Stock Exchange)<br>
        - Operating company: Intercontinental Exchange (ICE)<br>
          https://www.theice.com/index
        - List of all stocks<br>
          https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download  
        - Directory of all stocks<br>
          download link: https://www.nyse.com/listings_directory/stock
    - NYSE American (known as `American Stock Exchange, AMEX` before 2017)
        - Operaing company: 
        - List of all stocks<br>
          download link: https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download
