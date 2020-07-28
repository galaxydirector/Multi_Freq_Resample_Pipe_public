import pandas as pd
import numpy as np
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm




class Preprocessor(object):
    
    """docstring for Preprocessor"""
    def __init__(self):
        pass

    def to_minute_data(self, configs, stock_data_df, include_otc=False):
        try:
            stock_data_df['DATETIME'] = pd.to_datetime(stock_data_df['DATETIME'], format='%Y-%m-%d %H:%M:%S')
            stock_data_df = stock_data_df.set_index('DATETIME')

        except KeyError as e:
            stock_data_df['index'] = pd.to_datetime(stock_data_df['index'], format='%Y-%m-%d %H:%M:%S')
            stock_data_df = stock_data_df.set_index('index')

        if not include_otc:
            stock_data_df = stock_data_df.between_time('9:30', "16:00")

        stock_price = stock_data_df['PRICE'].resample('1T').mean().fillna(method='ffill')
        trade_size = stock_data_df['SIZE'].resample('1T').sum().fillna(method='ffill')
        date = stock_data_df.index
        mapping = {"price":stock_price, "volume":trade_size, "date": date}

        columns = [mapping[feature] for feature in configs["data"]["features"]]

        # output is a numpy array, with one day data and all features
        return pd.concat(columns, axis=1).values.reshape(-1, len(configs["data"]["features"]))

    # def sliding_window(self, original_list, win_length, slide_step=1):
    #     """ Generic sliding window generator
    #     Current version does not perform any check, and
    #     assumes slide step is 1
        
    #     original_list: float[][][]
    #     win_length: int
    #     output: float[sliding window][feature][daily df], e.g. float[day 1][price df, volume df]
    #     """
    #     res = []
    #     for i in range(0, len(original_list)-win_length+1):
    #         window = original_list[i: i+win_length]

    #         if len(window) > 0:
    #             res.append(window) 

    #     return res


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

        Caution: This operation requires close price at column 0 in df!!!
        
        args:
        data_window: data_window[daily][df]

        df has dimension of (time_series_steps, num_features)

        output: 
        float[] a long series of the whole year transformation
        """
        
        output = []
        for prev_day, one_day in zip(data_window, data_window[1:]):
            prev_close = prev_day[-1][0]

            for row in one_day:
                temp = []
                # row[0] is a hyper param, which price needs to be the first feature
                temp.append(self.log_return(row[0],prev_close)) 
                # deep copy rest of features into matrix
                for i in range(1,len(row)):
                    temp.append(row[i])

                # put every minute into the output
                output.append(temp)
            # output.extend([self.log_return(now, prev_close) for now in one_day])

        return output
