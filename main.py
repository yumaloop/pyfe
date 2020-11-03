import datetime
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader.stooq as stooq

# ソニー（SNE）の株価情報
# df_sne = web.DataReader('SNE', 'yahoo', start, end)

# https://stooq.com/
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2020, 11, 30)
stockcode="6701"
df = stooq.StooqDailyReader(f"{stockcode}.jp", start, end).read()
df = df.sort_values(by='Date',ascending=True)
print(df)
