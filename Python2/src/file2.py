import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tushare as ts

sns.set_style("ticks")  # sns选择样式
path = os.path.dirname(__file__)


class Wavelet(object):
    def __init__(self, path=path):
        self.path = path

    def get_tushare_k_data(self, code='002237', csv_file_name='\\file\\file2.csv'):
        csv_file = self.path + csv_file_name
        df = ts.get_k_data(code)
        try:
            df.to_csv(csv_file, index=False)
            print('writing csv file:\n', csv_file)
        except Exception as e:
            print(e)

    def read_csv_file(self, csv_file_name='\\file\\file2.csv'):
        csv_file = self.path + csv_file_name
        # csv源文件是没有数据类型的，读取的时候直接指定类型读取
        df = pd.read_csv(csv_file,
                         dtype={'open': np.float32,  # 开盘价：浮点型
                                'high': np.float32,
                                'low': np.float32,
                                'close': np.float32,
                                'volume': np.int32,  # 成交量：整型
                                'code': str},  # 代码：字符串
                         parse_dates=['date'],
                         date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d'))  # 日期特别处理

        with sns.axes_style("darkgrid"):
            x, y = df['date'], df['open']
            z = y * 0.9
            plt.plot(x, y)
            plt.fill_between(x.values, y.values, z, facecolor='c')
            plt.scatter(x[y.argmax(axis=1, skipna=True)],
                        y.max(),
                        facecolor='None',
                        edgecolor='r',
                        linewidth=2,
                        s=200)
            plt.annotate('Max',
                         xy=(x[y.argmax(axis=1, skipna=True)],
                             y.max()),
                         xytext=(x[y.argmax(axis=1, skipna=True) + 50],
                                 y.max() * 0.9),
                         arrowprops=dict(arrowstyle="->", connectionstyle="angle3", facecolor="r"))
            plt.title('002237', fontsize='xx-large', fontweight='bold')
            plt.legend(['open'])
            plt.xlabel('date')
            plt.ylabel('open')
            plt.tight_layout()
            plt.show()

    def run(self):
        self.read_csv_file()


w = Wavelet()
w.run()
