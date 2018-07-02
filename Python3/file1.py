# coding:utf-8
import os
import time

from plotly.grid_objs import Grid, Column
import plotly

import datetime as dt
import pandas as pd
import plotly.plotly as py
import tushare as ts


plotly.offline.init_notebook_mode(connected=True)


class PlotlyAnimation(object):
    def __init__(self):
        self.run()

    def get_csv_data(self, path='\\file\\', file='file1.csv'):
        # 读取csv路径
        self.filepath = os.path.dirname(__file__) + path  # csv文件所在的文件夹
        if not os.path.exists(self.filepath):  # 目录不存在则创建文件夹
            os.system('mkdir ' + self.filepath)
        # 读取csv数据
        csvfile = self.filepath + file
        if not os.path.exists(csvfile):  # csv不存在则从tushare下载
            df = ts.get_k_data('002237', start='2010-01-01', end='2018-01-01')
            df.to_csv(csvfile, index=False)  # index设为False
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d')  # 日期类型转换
            )
            return df
        else:  # 有则读取
            df = pd.read_csv(csvfile)
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d')  # 日期类型转换
            )
            return df

    def make_the_grid(self,):
        df = self.get_csv_data()
        close = list(df['close'])
        columns = []
        self.n = len(close)
        for i in range(1, self.n):
            x = Column(list(df.index.values)[:i], 'x{}'.format(i - 1))
            columns.append(x)
            y = Column(close[:i], 'y{}'.format(i - 1))
            columns.append(y)
        grid = Grid(columns)
        return grid

    def make_the_figure(self):
        def to_unix_time(self, dtime):
            epoch = dt.datetime.utcfromtimestamp(0)
            return (dtime - epoch).total_seconds() * 1000

        grid = self.make_the_grid()
        data = [dict(type='scatter',
                     xsrc=grid.get_column_reference('x1'),
                     ysrc=grid.get_column_reference('y1'),
                     name='close',
                     mode='lines',
                     line=dict(color='rgb(114, 186, 59)'),
                     fill='tozeroy',
                     fillcolor='rgba(114, 186, 59, 0.5)')]

        axis = dict(ticklen=4,
                    mirror=True,
                    zeroline=False,
                    showline=True,
                    autorange=False,
                    showgrid=False)

        layout = dict(title='002237',
                      font=dict(family='Balto'),
                      showlegend=False,
                      autosize=False,
                      width=800,
                      height=400,
                      xaxis=dict(axis, **{'nticks': 12, 'tickangle': -45,
                                          'range': [to_unix_time(self, dt.datetime(2010, 1, 1)),
                                                    to_unix_time(self, dt.datetime(2017, 1, 1))]}),
                      yaxis=dict(axis, **{'title': '$', 'range': [0, 120]}),
                      updatemenus=[dict(type='buttons',
                                        showactive=False,
                                        y=1,
                                        x=1.1,
                                        xanchor='right',
                                        yanchor='top',
                                        pad=dict(t=0, r=10),
                                        buttons=[dict(label='Play',
                                                      method='animate',
                                                      args=[None, dict(frame=dict(duration=50, redraw=False),
                                                                       transition=dict(
                                                                           duration=0),
                                                                       fromcurrent=True,
                                                                       mode='immediate')])])])

        frames = [{'data': [{'xsrc': grid.get_column_reference('x{}'.format(i - 1)),
                             'ysrc': grid.get_column_reference('y{}'.format(i - 1))}],
                   'traces': [0]
                   } for i in range(1, self.n)]

        fig = dict(data=data, layout=layout, frames=frames)
        py.icreate_animations(fig, 'AAPL-stockprice' + str(time.time()))

    def run(self):
        self.make_the_figure()


pa = PlotlyAnimation()
