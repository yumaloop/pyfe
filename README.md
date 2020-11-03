# afe2_backtest

```
# install pandas-datareader
pip install git+https://github.com/pydata/pandas-datareader.git
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
