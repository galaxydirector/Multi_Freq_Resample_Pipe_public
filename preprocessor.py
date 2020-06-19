import pandas as pd
import numpy as np
import datetime


class Preprocessor(object):
    
    """docstring for Preprocessor"""
    def __init__(self):
        pass

    # TODO: deprecated
    def to_minute_data(self, configs, stock_data_df, include_otc=False):
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

        mapping = {"price":stock_price, "volume":trade_size, "date": date}

        columns = [mapping[feature] for feature in configs["data"]["features"]]
        # print(columns)
        # output is a numpy array, with one day data and all features
        return pd.concat(columns, axis=1).values.reshape(-1, len(configs["data"]["features"]))
        
    
    def groupby_time(self,configs,file_system_df_list,time_range,method,include_otc=False):
        """
        stock_data_dfs: list of dfs
        """
        stock_data_dfs = [self.__datetime_format_process__(i) for i in file_system_df_list]

        if method == 'day':
            i = 0
            res = []
            while i < len(stock_data_dfs):
                tmp = stock_data_dfs[i:i+time_range]
                res.append(self.__day_range__(configs,tmp,time_range,include_otc=False))
                i = i + time_range

            return res

        elif method == 'minute':
            return [self.__minute_range__(configs,df,time_range,include_otc) for df in stock_data_dfs]

        elif method == 'hour':
            time_range = 60 ### TODO: Double check
            return [self.__minute_range__(configs,df,time_range,include_otc) for df in stock_data_dfs]

    def __minute_range_helper__(self,stock_data_df,time_range,include_otc=False):
        '''
        time_range: int, the time range in minutes
        '''
        try:
            stock_data_df['DATETIME'] = stock_data_df['DATETIME']
            stock_data_df = stock_data_df.set_index('DATETIME')
        except KeyError as e:
            
            stock_data_df['index'] = pd.to_datetime(stock_data_df['index'], format='%Y-%m-%d %H:%M:%S')
            stock_data_df = stock_data_df.rename(columns = {'index':'DATETIME'})
            stock_data_df['DATETIME'] = stock_data_df['DATETIME']
            stock_data_df = stock_data_df.set_index('DATETIME')

        if not include_otc:
            stock_data_df = stock_data_df.between_time('9:30', "15:59")  

        new_df = stock_data_df.groupby(pd.Grouper(level='DATETIME',freq= str(time_range) + 'min')).mean().fillna(method='ffill')[['PRICE']]
        new_df['SIZE'] = stock_data_df.groupby(pd.Grouper(level='DATETIME',freq= str(time_range) + 'min')).sum().fillna(method='ffill')['SIZE']
        # TODO: more features: low, high, close, open
        new_df['low'] = stock_data_df.groupby(pd.Grouper(level='DATETIME',freq= str(time_range) + 'min')).max().fillna(method='ffill')['PRICE']
        new_df['high'] = stock_data_df.groupby(pd.Grouper(level='DATETIME',freq= str(time_range) + 'min')).min().fillna(method='ffill')['PRICE']
        new_df['open'] = stock_data_df.groupby(pd.Grouper(level='DATETIME',freq= str(time_range) + 'min')).first().fillna(method='ffill')['PRICE']
        new_df['close'] = stock_data_df.groupby(pd.Grouper(level='DATETIME',freq= str(time_range) + 'min')).last().fillna(method='ffill')['PRICE']

        new_df = new_df.reset_index()
        new_df['DATETIME'] = new_df['DATETIME']
        new_df.set_index('DATETIME')
        # print(new_df)
        return new_df
    

    def __minute_range__(self,configs,stock_data_df,time_range,include_otc=False):
        '''
        configs: the loaded config
        stock_data_dfs: list, list of dataframe
        time_range: int, number of minutes
        '''
        new_df = self.__minute_range_helper__(stock_data_df,time_range,include_otc=False)
        features = configs["data"]["features"]
        mapping = {"price":"PRICE", "volume":"SIZE",'date':'DATETIME'}
        return new_df[[mapping[feature] for feature in features]].values.reshape(-1, len(features))



    def __day_range__(self,configs,stock_data_dfs,time_range,include_otc=False):
        '''
        configs: the loaded config
        stock_data_dfs: list, list of dataframe
        time_range: int, number of days 
        '''
        if time_range == 1:
            # TODO: magic number?
            # solve in straight forward way
            new_df = self.__minute_range_helper__(stock_data_dfs[0],24*60,include_otc=False)

        else:
            tmp_dfs = stock_data_dfs
            tmp_df = tmp_dfs[0]
                
            for j in range(1,len(tmp_dfs)):
                tmp_df = pd.concat([tmp_df,tmp_dfs[j]])

            # TODO: magic number?
            # make sure of the right way to group by
            new_df = self.__minute_range_helper__(tmp_df,24*60*365,include_otc=False)

        
        features = configs["data"]["features"]
        mapping = {"price":"PRICE", "volume":"SIZE",'date':'DATETIME'}
        return new_df[[mapping[feature] for feature in features]].values.reshape(-1, len(features))
    
    # def __datetime_format_process__(self, df):
    #     try:    
    #         col_test = len(df['DATETIME'])
    #     except KeyError as e:
    #         df = df.rename(columns={'index':'DATETIME'})
    #     if type(df['DATETIME'][0]) == str:
    #         df['DATETIME'] = df['DATETIME'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    #     return df

    def __datetime_format_process__(self,df):
        try:    
            col_test = len(df['DATETIME'])
        except KeyError as e:
            df = df.rename(columns={'index':'DATETIME'})
        if type(df['DATETIME'][0]) == str:
            df['DATETIME'] = df['DATETIME'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
        return df


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

    def batch_log_transform_for_test(self, data_window):
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
        logs = []
        times = []
        ori_price = []
        prev_close = -1

        

        for prev_day, one_day in zip(data_window, data_window[1:]):
            prev_close = prev_day[-1][0]
            for row in one_day:
                temp = []
                # row[0] is a hyper param, which price needs to be the first feature
                temp.append(self.log_return(row[0],prev_close)) 
                logs.append(self.log_return(row[0],prev_close))
                times.append(row[-1])
                ori_price.append(row[0])
                # deep copy rest of features into matrix
                for i in range(1,len(row)):
                    temp.append(row[i])

                # put every minute into the output
                output.append(temp)
            # output.extend([self.log_return(now, prev_close) for now in one_day])

        return output,prev_close,logs,ori_price,times

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