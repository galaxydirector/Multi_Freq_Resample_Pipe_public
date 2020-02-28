import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np



class Preprocessor(object):
    
    """docstring for Preprocessor"""
    def __init__(self):
        pass

    def to_minute_data(self, stock_data_df, include_otc=False):
        try:
            stock_data_df['DATETIME'] = pd.to_datetime(stock_data_df['DATETIME'], format='%Y-%m-%d %H:%M:%S')
            stock_data_df = stock_data_df.set_index('DATETIME')

        except KeyError as e:
            stock_data_df['index'] = pd.to_datetime(stock_data_df['index'], format='%Y-%m-%d %H:%M:%S')
            stock_data_df = stock_data_df.set_index('index')

        if not include_otc:
            stock_data_df = stock_data_df.between_time('9:30', "15:59")

        stock_price = stock_data_df['PRICE'].resample('1T').mean().fillna(method='ffill')
        trade_size = stock_data_df['SIZE'].resample('1T').sum().fillna(method='ffill')
        date = stock_data_df.index
        # price = stock_price

        return {"price":stock_price, "vol":trade_size, "date":date}

    def sliding_window(self, original_list, win_length, slide_step=1):
        """ Generic sliding window generator
        Current version does not perform any check, and
        assumes slide step is 1
        
        original_list: float[][]
        win_length: int
        output: float[][]
        """
        for i in range(0, len(original_list)-win_length+1):
            window = original_list[i: i+win_length]

            if len(window) > 0:
                yield window



    def log_return(self, a, b):
        """
        a,b: float
        output: log(a/b)
        """
        return np.log(a)-np.log(b)

    def batch_log_transform(self, data_window):
        """
        This transformation uses log return for each minute 
        comparing to previous day close price
        
        args:
        data_window: float[][df]

        output: 
        float[] a long series of the whole year transformation
        """
        output = []

        # convert pd into np
        for i in range(len(data_window)):
            try:
                merged_np = pd.concat(data_window[i]).values.reshape(-1, 1)
                data_window[i] = merged_np
            except TypeError as e:
                raise e

        for prev_day, one_day in zip(data_window, data_window[1:]):
            prev_close = prev_day[-1]

            output.extend([self.log_return(now, prev_close) for now in one_day])

        return output
