# coding:utf-8
from matplotlib.gridspec import GridSpec
import pywt

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
做多振幅：给定时间窗口的收盘价的振幅
做多波动率：给定时间窗口的对数收益率的标准差
"""

df = pd.read_csv('e:\\002237.csv')[['date', 'close']]

df['log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
df['volatility_20'] = df['log_ret'].rolling(window=20).std()
df['volatility_240'] = df['log_ret'].rolling(window=120).std()
df['swing_20'] = df['close'].rolling(
    window=20).max() - df['close'].rolling(window=20).min()
df['swing_240'] = df['close'].rolling(
    window=240).max() - df['close'].rolling(window=240).min()

df['date'] = df['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
df.dropna(inplace=True)

d = df['date'].values
c = df['close'].values
r = df['log_ret'].values

v_20 = df['volatility_20'].values
v_240 = df['volatility_240'].values
v_20_wt = pywt.idwt(pywt.dwt(v_20, 'db2')[0], None, 'db2')
v_240_wt = pywt.idwt(pywt.dwt(v_240, 'db2')[0], None, 'db2')

s_20 = df['swing_20'].values
s_240 = df['swing_240'].values
s_20_wt = pywt.idwt(pywt.dwt(s_20, 'db2')[0], None, 'db2')
s_240_wt = pywt.idwt(pywt.dwt(s_240, 'db2')[0], None, 'db2')

fig = plt.figure(1)
gs = GridSpec(3, 1)

plt.style.use('seaborn-dark-palette')
ax1 = plt.subplot(gs[0])
ax1.plot(d, c, color="black")
ax1.bar(d, r, color="r")
ax1.legend(['close', 'log_ret'])
plt.ylabel('close')
plt.title('002237', fontsize=16)

ax2 = plt.subplot(gs[1])
ax2.plot(d, v_20_wt[:len(d)])
ax2.plot(d, v_240_wt[:len(d)])
ax2.bar(d, v_20_wt[:len(d)] - v_240_wt[:len(d)], color='red')
ax2.legend(['month(20)', 'year(240)'])
plt.ylabel('volatility')

ax3 = plt.subplot(gs[2])
ax3.plot(d, s_20_wt[:len(d)])
ax3.plot(d, s_240_wt[:len(d)])
ax3.legend(['month(20)', 'year(240)'])
plt.xlabel('date')
plt.ylabel('swing')

plt.show()
