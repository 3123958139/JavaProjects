# coding:utf-8
from matplotlib.gridspec import GridSpec
import pywt

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""

"""

df = pd.read_csv('e:\\002237.csv')[['date', 'close']]

df['log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
df['volatility_20'] = df['log_ret'].rolling(window=20).std()
df['volatility_240'] = df['log_ret'].rolling(window=120).std()

df['date'] = df['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
df.dropna(inplace=True)

d = df['date'].values
c = df['close'].values
r = df['log_ret'].values
v_20 = df['volatility_20'].values
v_240 = df['volatility_240'].values
v_20_wt = pywt.idwt(pywt.dwt(v_20, 'db2')[0], None, 'db2')
v_240_wt = pywt.idwt(pywt.dwt(v_240, 'db2')[0], None, 'db2')

fig = plt.figure(1)
gs = GridSpec(3, 1)

plt.style.use('seaborn-dark-palette')
ax1 = plt.subplot(gs[:2])
ax1.plot(d, c, color="black")
ax1.bar(d, r, color="r")
ax1.legend(['close', 'log_ret'])
plt.title('002237', fontsize=16)
plt.ylabel('close')
ax2 = plt.subplot(gs[2])
ax2.plot(d, v_20_wt[:len(d)])
ax2.plot(d, v_240_wt[:len(d)])
ax2.bar(d, v_20_wt[:len(d)] - v_240_wt[:len(d)], color='red')
ax2.legend(['month(20)', 'half of year(240)'])
plt.xlabel('date')
plt.ylabel('volatility')
plt.show()
