# coding:utf-8
import os

from matplotlib import rcParams

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tushare as ts


class Quant(object):
    #=========================================================================
    #
    #=========================================================================
    def __init__(self, read=True):
        self.read = read
        self.path = os.path.dirname(__file__)
        pass

    #=========================================================================
    # 历史数据
    #=========================================================================
    def TSData(self, csv='file2.csv'):
        #
        csv = self.path + '\\file2_pics\\' + csv
        #
        if self.read == False:
            df = ts.get_k_data('002237')
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            df.to_csv(csv, index=False)
            return df

        else:
            df = pd.read_csv(csv)
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            return df

    #=========================================================================
    # 最大上涨、最大回撤
    #=========================================================================
    def MaxGrowUpDrawDown(self, series=None):
        #
        ui = np.argmax(series - np.minimum.accumulate(series))
        uj = np.argmin(series[:ui])
        max_grow_up = (float(series[ui]) / series[uj]) - 1.
        max_grow_up = round(max_grow_up / (ui - uj) * 365, 2)
        #
        di = np.argmax(np.maximum.accumulate(series) - series)
        dj = np.argmax(series[:di])
        max_draw_down = (float(series[di]) / series[dj]) - 1.
        max_draw_down = round(max_draw_down / (di - dj) * 365, 2)
        return ui, uj, max_grow_up, di, dj, max_draw_down

    #=========================================================================
    # 全局最大、全局最小
    #=========================================================================
    def ArgMaxMin(self, series=None):
        argmax, max = series.argmax(), series.max()
        argmin, min = series.argmin(), series.min()
        return argmax, max, argmin, min

    #=========================================================================
    # 年化收益率、夏普比率
    #=========================================================================
    def RateShape(self, series=None):
        rate = (series[-1] - series[0])/(series.index[-1]-series.index[0])*365
        rate_log=series-series.diff(1)
        shape=

    #=========================================================================
    # 画图
    #=========================================================================
    def Plot(self, feeddict={}):
        #
        date = feeddict['date']
        open = feeddict['open']
        ui = feeddict['ui']
        uj = feeddict['uj']
        print('最长上涨天数：', (ui - uj))
        max_grow_up = feeddict['max_grow_up']
        print('期间收益率：', max_grow_up * 100, '%')
        di = feeddict['di']
        dj = feeddict['dj']
        print('最长下跌天数：', (di - dj))
        max_draw_down = feeddict['max_draw_down']
        print('期间收益率：', max_draw_down * 100, '%')
        argmax = feeddict['argmax']
        max = feeddict['max']
        argmin = feeddict['argmin']
        min = feeddict['min']
        print('全期间最大：', max, '全期间最小：', min)
        #
        plt.plot(date, open, color='black')
        #
        plt.plot([date[ui], date[uj]], [open[ui], open[uj]], color='r')
        plt.text(date[ui],
                 open[ui],
                 ' ' + str(max_grow_up * 100) + '%',
                 color='r')
        plt.plot([date[di], date[dj]], [open[di], open[dj]], color='g')
        plt.text(date[di],
                 open[di],
                 ' ' + str(max_draw_down * 100) + '%',
                 color='g')
        #
        plt.scatter(date[argmax], max,
                    facecolor='None', edgecolor='red', linewidths=2, s=200)
        plt.scatter(date[argmin], min,
                    facecolor='None', edgecolor='green', linewidths=2, s=200)
        #
        plt.xlabel('date')
        plt.ylabel('open')
        #
        plt.title('002237',
                  fontdict={'fontsize': 18, 'fontweight': rcParams['axes.titleweight']})
        #
        plt.show()

    #=========================================================================
    #
    #=========================================================================
    def Run(self):
        #
        feeddict = {}
        #
        df = self.TSData()
        feeddict['date'] = df['date']
        feeddict['open'] = df['open']
        feeddict['ui'], feeddict['uj'], feeddict['max_grow_up'],\
            feeddict['di'], feeddict['dj'], feeddict['max_draw_down'] \
            = self.MaxGrowUpDrawDown(df['open'])
        feeddict['argmax'], feeddict['max'], feeddict['argmin'], feeddict['min'] \
            = self.ArgMaxMin(df['open'])
        #
        self.Plot(feeddict)


q = Quant()
q.Run()
