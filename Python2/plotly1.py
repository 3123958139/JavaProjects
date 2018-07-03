# coding:utf-8
'''
reference:https://plot.ly/python/
'''
import datetime as dt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import sklearn.preprocessing as pc
import tushare as ts


#=========================================================================
# 使用plotly的离线模式
#=========================================================================
py.init_notebook_mode(connected=True)  # plotly的离线模式，不需要登录plotly帐号
#=========================================================================
# 准备交互图表的数据
#=========================================================================
# df = ts.get_k_data('002237')  # 从tushare下载股票日线数据
df = pd.read_csv(
    'D:\\Program Files\\eclipse-cpp-oxygen-3a-win32-x86_64\\tmp\\JavaProjects\\Python2\\src\\file\\file2.csv')
#
date = df['date'].map(
    lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))  # 日期由str类型修改datetime
high = df['high']
volume = df['volume']
# 标准化
high_scaled = pc.scale(high)
volume_scaled = pc.scale(volume)
#
date = list(date)
high_scaled = list(high_scaled)
volume_scaled = list(volume_scaled)
#
trace1 = go.Scatter(x=date, y=high_scaled, name='high')
trace2 = go.Bar(x=date, y=volume_scaled, name='volume')
#
data = [trace1, trace2]  # 交互图表的数据
#=========================================================================
# 交互图表布局的定义
#=========================================================================
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
#=========================================================================
# 执行交互图表
#=========================================================================
fig = dict(data=data, layout=layout)
py.plot(fig)
