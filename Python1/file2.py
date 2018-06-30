# coding:utf-8
import os

from matplotlib import rcParams

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tushare as ts
#=========================================================================
# 图表显示中文
#=========================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


#=========================================================================
# 量化回测模块
#=========================================================================
class Quant(object):
    #=========================================================================
    # 一些需提供的全局变量放在这里初始化
    #=========================================================================
    def __init__(self, read=True):
        self.read = read  # 控制是从tushare下数据还是从本地csv读数据
        self.path = os.path.dirname(__file__)  # 当前py文件的目录

    #=========================================================================
    # tushare历史数据，先下载保存到csv，再从csv读取数据，注意日期的调整
    #=========================================================================
    def TSData(self, csv='file2.csv'):
        #
        csv = self.path + '\\file2_pics\\' + csv  # 指定csv的存取路径
        #
        if self.read == False:
            df = ts.get_k_data('002237')  # 从tushare下载数据
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))  # 日期格式的调整
            df.to_csv(csv, index=False)  # 保存csv，注意index=False
            return df

        else:
            df = pd.read_csv(csv)  # 读取已有csv数据
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))  # 依然需要调整日期格式
            return df

    #=========================================================================
    # 最大上涨、最大回撤
    #=========================================================================
    def MaxGrowUpDrawDown(self, series=None):
        # 最大上涨
        ui = np.argmax(series - np.minimum.accumulate(series))
        uj = np.argmin(series[:ui])
        max_grow_up = (float(series[ui]) / series[uj]) - 1.
        max_grow_up = round(max_grow_up / (ui - uj) * 365, 2)  # 年化处理
        # 最大回撤
        di = np.argmax(np.maximum.accumulate(series) - series)
        dj = np.argmax(series[:di])
        max_draw_down = (float(series[di]) / series[dj]) - 1.
        max_draw_down = round(max_draw_down / (di - dj) * 365, 2)  # 年化处理
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
        p = series.values  # 价格
        n = series.index.values  # 索引
        rate = (p[-1] - p[0]) / p[0] / (n[-1] - n[0]) * 365  # 年化收益率
        rate = round(rate, 2)
        ret = series.map(lambda x: np.log(x)) - \
            series.map(lambda x: np.log(x)).shift(1)  # log收益率序列
        shape = ret.mean() / ret.std()  # 夏普比率
        shape = round(shape, 2)
        return rate, shape

    #=========================================================================
    # 画图
    #=========================================================================
    def Plot(self, feeddict={}):
        #
        date = feeddict['date']
        open = feeddict['open']
        # 最大上涨、最大下跌
        ui = feeddict['ui']
        uj = feeddict['uj']
        max_grow_up = feeddict['max_grow_up']
        print('最长上涨天数：', (ui - uj),
              '\t期间年化收益率：', max_grow_up * 100, '%',
              '\t起止日期：', date[uj], ' - ', date[ui])
        di = feeddict['di']
        dj = feeddict['dj']
        max_draw_down = feeddict['max_draw_down']
        print('最长下跌天数：', (di - dj), '\t期间年化收益率：', max_draw_down * 100, '%',
              '\t起止日期：', date[dj], ' - ', date[di])
        # 最大、最小
        argmax = feeddict['argmax']
        max = feeddict['max']
        argmin = feeddict['argmin']
        min = feeddict['min']
        print('全期间最大价格：', max, '\t发生日期：', date[argmax])
        print('全期间最小价格：', min, '\t发生日期：', date[argmin])
        # 年化收益率、夏普比率
        rate, shape = self.RateShape(open)
        print('全日期年化收益率：', rate * 100, '%\t全日期夏普比率：', shape,
              '\t起止日期：', date[0], ' - ', date[date.index.values[-1]])
        # 主图
        grid = plt.GridSpec(6, 1, wspace=0.5, hspace=0.5)
        plt.subplot(grid[:4])
        plt.plot(date, open, color='black')
        # 最大上涨、最大回撤
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
        # 最大值、最小值
        plt.scatter(date[argmax], max,
                    facecolor='None', edgecolor='red', linewidths=2, s=200)
        plt.scatter(date[argmin], min,
                    facecolor='None', edgecolor='green', linewidths=2, s=200)
        #
        plt.ylabel('开盘价')
        #
        plt.text(date[di], int((max + min) / 2),
                 '\n年化收益率【' + str(rate * 100) + '%】\n' +
                 '夏普比率【' + str(shape) + '】',
                 color='blue')
        plt.title('002237',
                  fontdict={'fontsize': 18, 'fontweight': rcParams['axes.titleweight']})
        #
        plt.subplot(grid[4:6])
        plt.plot(date, np.maximum.accumulate(open) - open, color='g')
        plt.plot(date, open - np.minimum.accumulate(open), color='r')
        plt.grid(b=None)
        plt.legend(['期间最大跌幅', '期间最大涨幅'])
        plt.xlabel('日期')
        plt.ylabel('期间最大涨幅\跌幅')
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
