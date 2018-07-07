import os

import pywt

import datetime as dt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po


csvdir = os.path.dirname(__file__)
csvfile = csvdir + '\\csv\\file1.csv'
df = pd.read_csv(csvfile)[['date', 'close']]

date = df['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
close = df['close']

close_cA, close_cD = pywt.dwt(close, 'db2')
close_dwt = pywt.idwt(close_cA, None, 'db2')

po.init_notebook_mode(connected=True)

trace_close = go.Scatter(
    x=date,
    y=close,
    name='close'
)
trace_close_dwt = go.Scatter(
    x=date,
    y=close_dwt[:len(close)],
    name='close_dwt'
)
data = [trace_close, trace_close_dwt]

layout = dict(
    title='002237 - made by dch',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='YTD',
                     step='year',
                     stepmode='todate'),
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
po.plot(fig)
