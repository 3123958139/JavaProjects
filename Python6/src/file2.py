# coding:utf-8

"""
投资组合优化
1.数据预处理
2.投资组合可行域
3.投资组合有效边界
4.
"""

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
import tushare as ts

#=========================================================================
# 数据准备
#=========================================================================


def DownloadTSData():
    """
    usage：从tushare下载5个股票日线数据
    """
    # 合并DataFrame时用于对齐的日期列，
    # 由于沪深300指数的日期是无间断的，可用沪深300的日期列作为标准。
    hs300_date = sorted(set(ts.get_k_data('399300')['date'].values))
    data = pd.DataFrame({'date': hs300_date})

    # 要下载数据的5个股票。
    symbols = ['600000', '600010', "600015", '600016', '600018']

    for sym in symbols:
        df = ts.get_k_data(sym)[['date', 'close']]
        df.rename(columns={'close': sym}, inplace=True)

        # 日期必须唯一，出现重复项报错。
        if len(set(df['date'].values)) != len(df['date']):
            raise 'date\'s error'

        # 只取日期和收盘价，左键合并保证日期列的完整与唯一。
        data = pd.merge(data, df, on='date', how='left')

    # 有些个股数据有缺失，用前值填充。
    data.fillna(method='ffill', inplace=True)

    # 本地持久化。
    data.to_csv('d:\\CAPM5.csv', index=False)


def ReadCSVData():
    """
    usage：读取本地的CSV数据
    """
    df = pd.read_csv('d:\\CAPM5.csv')

    # 从CSV读取的数据是str格式的，需转为datetime64格式。
    df['date'] = df['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

    # 将日期作为索引列。
    df.set_index(keys='date', inplace=True)

    cols = df.columns

    # 首日对齐，计算累积收益率，
    for col in cols:
        df[col + '1'] = df[col] / df[col].ix[0] * 100

    # 计算log收益率备用。
    for col in cols:
        df[col + '2'] = np.log(df[col] / df[col].shift(1)) * 100

    # 计算log收益率后第一行是NaN值，去掉
    df.fillna(value=0, inplace=True)
    return df


#=========================================================================
# 投资组合可行域
#=========================================================================
df = ReadCSVData()

# # 5个5个画图
# df.iloc[:, :5].plot()
# df.iloc[:, 5:10].plot()
# df.iloc[:, 10:].plot()
# plt.show()

# 选择收益率
rets = df.iloc[:, 10:]

# 计算收益率的均值
means = rets.mean()

prets = []
pvols = []
noa = len(rets.columns)

for i in range(2500):
    # 生成随机权重并归一化
    weights = np.random.random(noa)
    weights /= np.sum(weights)

    # 权重确定后计算组合收益率均值和标准差
    mu = np.sum(means * weights) * 252
    sigma = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

    prets.append(mu)
    pvols.append(sigma)

# # 画出投资组合可行域
# plt.scatter(pvols, prets, marker='o')
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.show()

#=========================================================================
# 投资组合有效前沿
#=========================================================================


def statistics(weights):
    """
    usage：输入权重输出组合收益率、组合标准差、组合夏普比率
    """
    weights = np.array(weights)
    pret = np.sum(means * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])


def min_func_sharpe(weights):
    """
    usage：计算最大夏普比率，或者说最小化负的夏普比率
    """
    return -statistics(weights)[2]


def min_func_port(weights):
    return statistics(weights)[1]


def min_func_variance(weights):
    return statistics(weights)[l] ** 2


trets = np.linspace(0.0, 0.25, 50)
tvols = []
cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - trets},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in weights)
opts = sco.minimize(min_func_sharpe,
                    noa * [1. / noa, ], method='SLSQP',
                    bounds=bnds, constraints=cons)
optv = sco.minimize(min_func_variance,
                    noa * [1. / noa, ], method='SLSQP',
                    bounds=bnds, constraints=cons)
for tret in trets:
    res = sco.minimize(min_func_port, noa * [1. / noa, ],
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
    tvols.append(res['fun'])

tvols = np.array(tvols)

plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.scatter(tvols, trets, c=trets / tvols, marker='x')
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
plt.show()
