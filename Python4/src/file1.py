# coding:utf-8
import pywt
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

"""
小波变换去噪：
a. 适用环境：
- 非平稳
- 非线性
- 高信噪比
b. 去噪方法：
股票价格时间序列的噪音常体现在低频部分（每天都有的无规则的随机波动），主要就是去掉这部分
"""

df = pd.read_csv('D:\\Program Files\\eclipse-cpp-oxygen-3a-win32-x86_64' +
                 '\\tmp\\JavaProjects\\Python4\\others\\file1.csv')

d = df['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
x = df['open']

cA, cD = pywt.dwt(x, 'db4')
new_x = pywt.idwt(cA, None, 'db4')
new_x = pd.Series(data=new_x, name='new_open')
'''
>>>print('len(d)=', len(d),
      '\nlen(x)=', len(x),
      '\nlen(cA)=', len(cA),
      '\nlen(cD)=', len(cD),
      '\nlen(new_x)=', len(new_x))
Out[1]:     
len(d)= 641 
len(x)= 641 
len(cA)= 324 
len(cD)= 324 
len(new_x)= 642
'''
plt.plot(d, x)
plt.plot(d[:len(cA)], cA)
plt.plot(d[:len(cD)], cD)
plt.plot(d, new_x[:len(d)])
plt.legend(fontsize=16)
plt.show()
