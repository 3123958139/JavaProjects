import os

import datetime as dt
import pandas as pd
import tushare as ts


class PlotlyAnimation(object):
    def __init__(self):
        #
        self.dirname = os.path.dirname(__file__)
        self.filepath = self.dirname + '\\file\\'
        if not os.path.exists(self.filepath):
            os.system('mkdir ' + self.filepath)
        #
        self.run()

    def get_data_set(self):
        csvfile = self.filepath + 'file1.csv'
        if not os.path.exists(csvfile):
            df = ts.get_k_data('002237', start='2010-01-01', end='2018-01-01')
            df.to_csv(csvfile, index=False)
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d')
            )
            return df
        else:
            df = pd.read_csv(csvfile)
            df['date'] = df['date'].map(
                lambda x: dt.datetime.strptime(x, '%Y-%m-%d')
            )
            return df

    def run(self):
        df = self.get_data_set()
        print(df)


pa = PlotlyAnimation()
