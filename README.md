# afe2_backtest

py backtest repo

### pypi packages

```
pip3 install yfinance
pip3 install pandas_datareader
pip3 install cvxopt
```

# Data resources

### Index

link to get the list of ticker symbols:
- https://stooq.com/t/
- https://finance.yahoo.com/trending-tickers


- Japan
    - TOPIX500
        - `^TPX`(stooq)
    - TOPIX Core30
        - `^TPXC30`(stooq)
    - Nikkei 225 
        - `^N225`(yahoo), `^NKX`(stooq)
- United States
    - S&P 500 Index
        - `^GSPC`(yahoo), `^SPX`(stooq)
    - Dow Jones Industrial Average
        - `^DJI`(yahoo), `^DJI`(stooq)
    - NASDAQ-100
        - `^NDX`(yahoo), `^NDX`(stooq)
    - NASDAQ Composite Index
        - `^IXIC`(yahoo), `^NDQ`(stooq)

### Market data

- Japan
    - TSE (Tokyo Stock Exchange)<br>
        - Operating company: Japan Exchange Group, Inc. (JPX)<br>
          https://www.jpx.co.jp/
        - 統計情報(株式関連)<br>
          https://www.jpx.co.jp/markets/statistics-equities/index.html
        - Stock list: 東証上場銘柄一覧<br>
          https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
        - 株価指数
          - 株価指数ラインナップ https://www.jpx.co.jp/markets/indices/line-up/index.html
        - Risk-free rate<br>
          - 日本国債10年物利回り: 0.01% [日本国債・金利 - Bloomberg](https://www.bloomberg.co.jp/markets/rates-bonds/government-bonds/japan)
          - 無担保コールレート: 0.001% [コール市場関連統計(毎営業日) - 日本銀行](https://www3.boj.or.jp/market/jp/menu_m.htm)
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
    - NYSE MKT (also known as `American Stock Exchange, AMEX` before 2017)
        - Operating company: Intercontinental Exchange (ICE)<br>
          https://www.theice.com/index
        - List of all stocks<br>
          download link: https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download

# References

- https://investexcel.net/
- https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/#backtesting-on-real-market-data

