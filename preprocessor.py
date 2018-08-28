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
        price = stock_price

        return price


        # if len(stock_price) < 390:
        #     stock_price = pd.concat([stock_price, pd.Series([stock_price.iloc[-1]]*(390-len(stock_price)))])

        # month = np.array([d.date().month for d in stock_data_df.index])
        # weekday = np.array([d.date().weekday() for d in stock_data_df.index])
        # time = np.array([[d.time().hour, d.time().minute, d.time().second] for d in stock_data_df.index]).T
        # trade_size = stock_data_df['size'].values

        # return month, weekday, time, price, trade_size
        # print(price)

    def batch_transform(self, df_window, receptive_field, include_otc=False):
        data_batch = []

        for df_list in df_window:
            try:
                merged = pd.concat(df_list).values.reshape(-1, 1)
                transformed_data = self.transform_weekly_data(merged, receptive_field)
                data_batch.append(transformed_data)

            except TypeError as e:
                # print(df_list)
                raise e

        return data_batch


    def transform_weekly_data(self, data_set, receptive_field):
        """
        Sigmoid transformation transforms data into sigmoid space. Assume inputs are
        column vectors.
        
        Argument: 
        data_set is a (6) days min level data in a column vector
        receptive_field is how many mins would be dependent on
        """
        piece = data_set

        # main part removed 

        # return size would be (6.5*60*6, 1)
        # return np.array(normalized)
        return