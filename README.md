# pyfe

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
- [list of ticker symbols, stooq](https://stooq.com/t/)
- [list of ticker symbols, yahoo](https://finance.yahoo.com/trending-tickers)


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
          - 短期金利(overnight rate)
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

### Funds

- 国内株式
  - 日経投資信託ランキング (シャープレシオ)
    https://www.nikkei.com/markets/fund/ranking/?type=sharperatio&term=1y&category1=&page=1
- https://investexcel.net/
- https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/#backtesting-on-real-market-data

### Technical

- http://kabu.com/investment/guide/technical/01.html

# References

**Peer-reviewed Journals**
- [The Journal of Financial Data Science](https://jfds.pm-research.com/)
- [The Journal of Portfolio Management](https://jpm.pm-research.com/)
- [The Journal of Algorithmic Finance](http://www.algorithmicfinance.org/)
- [Cambridge University Press](https://www.cambridge.org/)

**Textbooks**
- [Investment Science](https://www.amazon.com/dp/0195108094), by David G. Luenberger, Oxford University Press (July 3, 1997)
- [金融工学入門 第2版 (日本語訳)](https://www.amazon.co.jp/dp/4532134587), by David G. Luenberger, 日本経済新聞出版; 第2版 (2015/3/26)
- [Advances in Financial Machine Learning](https://www.amazon.com/dp/1119482089), by Marcos Lopez de Prado, Wiley; 1st edition (February 21, 2018)
- [Machine Learning for Asset Managers](https://www.amazon.com/dp/1108792898), by Marcos M López de Prado, Cambridge University Press (April 30, 2020)

**Community**
- [Quants Portal](http://www.quantsportal.com/)

**Media**
- [seekingalpha.com](https://seekingalpha.com/)
- [talkmarkets.com](https://talkmarkets.com/)

