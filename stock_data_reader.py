# Stable version

# import re
import os
import os.path as path
import tensorflow as tf
import pandas as pd

import itertools
import numpy as np
import threading
import time
import datetime
import matplotlib.pyplot as plt

from Multi_Freq_Resample_Pipe_public.preprocessor import Preprocessor
from Multi_Freq_Resample_Pipe_public.db_manager import DBManager
print("StockDataReader Dependency Loaded")

DEBUG_ROOT_PATH = os.path.expanduser("~/Desktop/Time_Series/Database/")
# symbol_list = ['FB','JPM','XOM']
# 'FB','JPM','XOM','GOOG','BAC','AAPL','MSFT','AAPL', 'AMZN','JNJ','INTC','CVX','WFC','V','UNH','HD', 'PFE', 'CSCO', 'T'


class StockDataReader:
	def __init__(self, configs,
				data_dir,
				coord,
				symbol_list,
				year_range,
				symbol_first,
				data_win_len,
				receptive_field=None,
				queue_size=5):
		# system initialize
		self.db_manager = DBManager(data_dir)
		self.preprocessor = Preprocessor()
		self.configs = configs

		self.coord = coord
		self.threads = []

		# processing params
		self.data_dir = data_dir
		self.symbol_list = symbol_list
		self.year_range = year_range
		self.symbol_first = symbol_first
		self.data_win_len = data_win_len
		# deprecated
		self.receptive_field = receptive_field

		# queue setup
		# self.trans_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
		self.trans_queue = tf.queue.PaddingFIFOQueue(queue_size,
										 ['float32'],
										 shapes=[(None, len(self.configs["data"]["features"]))]) ### TODO: shape needs to be a variable

		# for multithreading:
		self.yield_list = itertools.product(self.symbol_list, self.year_range) if self.symbol_first else itertools.product(self.year_range, self.symbol_list)
		
	# deprecated
	def main_thread(self, sess):
		full_stock_data_list =[]
		full_sig_data_list = []

		# iter has to run before start thread
		iter_order = itertools.product(self.symbol_list, self.year_range) if self.symbol_first else itertools.product(self.year_range, self.symbol_list)

		# iterate through all symbol and years
		for iter_order_a, iter_order_b in iter_order:
			data_list = self.db_manager.get_unzipped_data(iter_order_a, iter_order_b)

			if data_list is not None: # need to check if it is under 6 days
				# convert from seconds to minutes
				resampled_data_list = [self.preprocessor.to_minute_data(data).get("price") for data in data_list]
				data_window = self.preprocessor.sliding_window(resampled_data_list, self.data_win_len)

				if data_window is not None:
					processed_data_window = self.preprocessor.batch_transform(data_window, self.receptive_field)
					for stock_data_list in processed_data_window:
						full_stock_data_list = np.append(full_stock_data_list, stock_data_list.reshape(-1, 1)[-390:])
			else:
				print("{} {} does not have data".format(iter_order_a, iter_order_b))

		return full_stock_data_list

	# deprecated 
	def dequeue_trans_many(self,num_elements):
		return self.trans_queue.dequeue_many(num_elements)

	def dequeue_trans(self):
		return self.trans_queue.dequeue()

	def main_thread_queue(self):
		stop = False

		"""Epochs
		In order to repeat whole dataset for several epoch, 
		whole list needs to repeatedly reconstructed.
		while and for loop is not allowed to enqueue, 
		otherwise same element would enqueue several times, 
		but cycle is a way to go around it"""
		iter_order = itertools.cycle(self.yield_list)

		# iterate through all symbol and years
		for iter_order_a, iter_order_b in iter_order:
			transition_data = self.db_manager.get_unzipped_data(symbol = iter_order_a, year = iter_order_b-1, last_few_days = self.data_win_len -1)
			data_list = self.db_manager.get_unzipped_data(symbol = iter_order_a, year = iter_order_b)
			
			# append the last few days of last year
			# check if any of the lists are empty
			if transition_data is not None and data_list is not None:
				new_data_list=transition_data+data_list
			elif transition_data is None and data_list is not None:
				new_data_list = data_list
			elif transition_data is not None and data_list is None:
				print("{} {} does not have data".format(iter_order_a, iter_order_b))
				continue
			elif transition_data is None and data_list is None:
				print("{} {} does not have data".format(iter_order_a, iter_order_b))
				continue

			if self.coord.should_stop():
				stop = True
				break

			# if data_list is not None: # need to check if it is under 6 days
			# convert from seconds to minutes
			# TODO: extend variable to multiple, price/volume: completed
			# resampled_data_matrix[daily][df]
			# resampled_data_matrix = [self.preprocessor.to_minute_data(self.configs, data) for data in new_data_list]
			resampled_data_matrix = self.preprocessor.groupby_time(self.configs, new_data_list, time_range=1, method='minute')

			# data_win_len is how many days to look at in one window
			# data_window = self.preprocessor.sliding_window(resampled_data_matrix, self.data_win_len)

			if resampled_data_matrix is not None:
				# this window is the whole year data, flatten into one array
				processed_data_window = self.preprocessor.batch_log_transform(resampled_data_matrix)

				# making sure the input into the queue has only one dimension
				# TODO: extend dimension to multiple: completed!
				assert np.array(processed_data_window).shape[1]==len(self.configs["data"]["features"])

				# determine the length
				# forecast_steps = self.configs["data"]["label_length"]
				# input_steps = self.configs["data"]["feature_length"]
				# total_len = input_steps+forecast_steps
				# assert len(processed_data_window)>=total_len

				# input the whole year
				self.trans_queue.enqueue((processed_data_window,))
				

				# # feed in the exact length to queue
				# for i in range(len(processed_data_window)-total_len+1):
				# 	self.trans_queue.enqueue((processed_data_window[i:i+total_len],))
					# sess.run(self.trans, feed_dict={self.trans_placeholder: processed_data_window[i:i+total_len]})


	def start_threads(self, n_threads=2):
		for _ in range(n_threads):
			thread = threading.Thread(target=self.main_thread_queue, args=())
			thread.daemon = True  # Thread will close when parent quits.
			thread.start()
			self.threads.append(thread)
			time.sleep(1)







class StockDataReaderForTest(StockDataReader):
	def __init__(self, configs,
				data_dir,
				coord,
				symbol_list,
				year_range,
				symbol_first,
				data_win_len,
				receptive_field=None,
				queue_size=5):
		
		super().__init__(configs,
				data_dir,
				coord,
				symbol_list,
				year_range,
				symbol_first,
				data_win_len,
				receptive_field=None,
				queue_size=5)
		self.trans_queue = None
		self.yield_list = []

	def __format_process__(self,df):
		try:    
			col_test = len(df['DATETIME'])
		except KeyError as e:
			df = df.rename(columns={'index':'DATETIME'})
		if type(df['DATETIME'][0]) == str:
			df['DATETIME'] = df['DATETIME'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
		return df

	def __search_specific_date__(self, array, t):
		'''
		Applied Binary Search
		input:
			array: the data list contains dataframes for each day
			t: the input date
		return:
			the index where the date locates
		'''
		#t = datetime.datetime.strptime(date,"%Y-%m-%d").date()
		low = 0
		height = len(array)-1
		while low <= height:
			mid = (low+height)//2
			#print(type(array[mid]))
			array[mid] = self.preprocessor.__datetime_format_process__(array[mid])
			# array[mid] = self.__format_process__(array[mid])

			if array[mid]['DATETIME'][0].date() < t:
				low = mid + 1

			elif array[mid]['DATETIME'][0].date() > t:
				height = mid - 1

			else:
				return mid

		return low

	def __search_date_period__(self, year, month, day, window, date_list):
		'''
		input:
			year: int, the year of input date
			month: int, the month of input date
			day: int, the day of input date
			window: int, the window size
			date_list: list of dataframes, each dataframe contains the data of a certain day.
						The list is sorted by date in ascending order.
						Each dataframe contains columns including DATETIME, PRICE, SIZE, SYMBOL
		return:
			res: list, a list of dataframes, 
				containing dataframe whose date range in [input date-window,input date] 
				each dataframe contains columns including DATETIME, PRICE, SIZE, SYMBOL
		'''
		date = datetime.date(year,month,day)
		res = []
		
		# date_list[0] = self.__format_process__(date_list[0])

		# l = date_list[0]['DATETIME'][0]

		# if date < l:
		# 	return res,date
			#raise Exception("OUT OF RANGE!")
		
		index = self.__search_specific_date__(date_list,date)
		
		date_list[index] = self.preprocessor.__datetime_format_process__(date_list[index])
		# date_list[index] = self.__format_process__(date_list[index])
		
		if type(date_list[index]) == str:
			date_list[index]['DATETIME'] = date_list[index]['DATETIME'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
		
		if date_list[index]['DATETIME'][0].date() != date:
			return res,date

		
		res = date_list[max(0,index - window):index+1]
		
		for i in range(len(res)):
			res[i] = self.preprocessor.__datetime_format_process__(res[i])
			# res[i] = self.__format_process__(res[i])
		return res,date

	def search_small_period(self,year,month,day,hour,minute,second,window,date_list):
		'''
		input:
			year : int , the year of a certrain time
			month : int , the month of a certrain time
			day : int, the day of a certrain time
			hour : int, the hour of a certrain time
			minute : int, the minute of a certrain time
			second : int, the second of a certrain time
			window: the window size [hours,minutes,seconds] = [24,25,18]
			date_list: list, a list of dataframe, each dataframe contains the data of a certrain day
						including DATETIME, PRICE, SIZE, SYMBOL,
						the dataframes are sorted according to date in asscending order.

		return:
			res: list, a list of dataframe, each dataframe contains the data of a certrain day
						including DATETIME, PRICE, SIZE, SYMBOL,
						the dataframes are sorted according to date in asscending order.
						All the data in the list locates between [input time-window,input time]
		'''
		res = []
		time = str(year)+'-'+str(month)+'-'+str(day)+' '+ str(hour) + ':'+ str(minute)+ ':'+ str(second)
		date = datetime.datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
		window_d = (window[0]/24 + window[1]/(24*60) + window[2]/(24*3600)) + 1
		if window_d < 100:
			window_d += 5
		else:
			window_d *= 1.05
		window_d = int(window_d)
		
		days,_ = self.__search_date_period__(year,month,day,window_d,date_list)
		#print('checkpoint1')
		if len(days) == 0:
			return [],date
		
		total_len = (self.configs["data"]["label_length"] + self.configs["data"]["feature_length"]) * 60
		i = len(days) - 1

		while i >= 0:
			day = days[i]
			if i == len(days) - 1:		
				tmp = day[day['DATETIME'] <= date]
				total_len -= len(tmp)
			else:
				total_len -= len(days[i])
			if total_len <=0:
				break
				
			i = i - 1
		#print('checkpoint2')
		if total_len > 0:
			return [],date
		
		res = days[i:]
		res[-1] = tmp
		# print('checkpoint3')
		return res,date
	
	def search_feature_length_period(self,year,month,day,hour,minute,second,date_list):
		date = datetime.date(year,month,day)
		res = []		
		index = self.__search_specific_date__(date_list,date)
		
		date_list[index] = self.preprocessor.__datetime_format_process__(date_list[index])	
		total_len = (self.configs["data"]["label_length"] + self.configs["data"]["feature_length"]) * 60
		i = index

		while i >= 0:
			day = date_list[i]
			if i == index:		
				tmp = day[day['DATETIME'] <= date]
				total_len -= len(tmp)
			else:
				total_len -= len(date_list[i])
			if total_len <=0:
				break
				
			i = i - 1
		#print('checkpoint2')
		if total_len > 0:
			return [],date
		
		res = date_list[i:index+1]
		res[-1] = tmp

		return res,date			

	def plt_plot(self,symbol,
						year,
						month,
						day,
						hour,
						minute,
						second,
						window, method, time_range):
		# TODO: generalize this?
		# TODO: rm all hard coding
		db_manager = DBManager("./Database/" + str(year) + "/" + symbol + "/",recursion_level=0)
		temp = db_manager.get_unzipped_data(symbol = symbol, year = year)
		data = []
		if month == 1: ### TODO: generalize, remove magic number
			# TODO: DB generalize
			try:
				db_manager1 = DBManager("./Database/" + str(year-1) + "/" + symbol + "/",recursion_level=0)	
				data.extend(db_manager.get_unzipped_data(symbol = symbol, year = year-1))
				data.extend(temp)
			except:
				data.extend(temp)
		else:
			pass
		res,input_date = self.search_small_period(year,
													month,
													day,hour,minute,second,
													window,data)
		resampled_data_matrix = self.preprocessor.groupby_time(self.configs,res,time_range,method)

		_,prev_close,logs,ori_prices,times = self.preprocessor.batch_log_transform_for_test(resampled_data_matrix)

		#logs
		plt.figure(1,figsize=(20,10))
		plt.title('logs')
		plt.plot(times,logs,'*')
		print(logs)

		#price
		plt.figure(2,figsize=(80,40))
		plt.title('price')
		plt.plot(ori_prices[:-1],'b*')
		plt.xticks(np.arange(len(times)-1), times)
		
		# plt.plot(times[-1],ori_prices[-1],'r*')
		# plt.plot(times[-1],ori_prices[-1]+0.05,'o')





def main_list():
	reader = StockDataReader(data_dir =DEBUG_ROOT_PATH,
				coord = [],
				symbol_list = symbol_list,
				year_range = range(2014, 2017),
				symbol_first = True,
				data_win_len = 6,
				receptive_field = 5*390+5)
	# full_stock_data_list, full_sig_data_list = reader.main_thread(sess=[])
	full_stock_data_list = reader.main_thread(sess=[])

	trans = full_stock_data_list.reshape(-1,1)
	np.savetxt(os.path.expanduser('~/Desktop/research/transformation/test/motherfucker_trans.csv'), np.array(trans),delimiter=',')

def queue_thread_main():
	coord = tf.train.Coordinator()
	reader = StockDataReader(data_dir = DEBUG_ROOT_PATH,
				coord = coord,
				symbol_list = symbol_list,
				year_range = range(2011, 2014),
				symbol_first = True,
				data_win_len = 6,
				receptive_field = 5*390+5)

	trans_full_list = []

	# operation
	trans_dequeue  = reader.dequeue_trans(1)

	session_config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 0,
									intra_op_parallelism_threads = 0)
	with tf.compat.v1.Session(config=session_config) as sess:
		threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
		reader.start_threads(sess)

		for _ in range(8): #19*6-2
			trans = sess.run(trans_dequeue)
			trans_full_list = np.append(trans_full_list, trans)

		coord.request_stop()
		coord.join(threads)


	trans = trans_full_list.reshape(-1,1)
	np.savetxt(os.path.expanduser('~/Desktop/research/transformation/test/tests.csv'), np.array(trans),delimiter=',')


def main_reader_for_test():
	coord = tf.train.Coordinator()
	session_config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 0,
								intra_op_parallelism_threads = 0)
	reader = StockDataReaderForTest(configs = session_config,
				data_dir = "Database/2014/FB/",
				coord = coord,
				symbol_list = ['FB'],
				year_range = range(2015, 2016),
				symbol_first = True,
				data_win_len = 6,
				receptive_field = 5*390+5)
	
	db_manager = DBManager("Database/2014/FB/",recursion_level=0)
	db_manager1 = DBManager("Database/2015/FB/",recursion_level=0)
	data = db_manager.get_unzipped_data(symbol = 'FB', year = 2014)
	data_2015 = db_manager1.get_unzipped_data(symbol = 'FB', year = 2015)
	data.extend(data_2015)
	a = time.time()
	res1 = search_date_period(2014,10,9,3,data)
	b = time.time()
	print(b-a)
	res2 = search_small_period(2015,5,11,9,30,1,[24,2,1],data)
	print(time.time()-b)
	print(res1)
	print("____________________________________")
	print(res2)



if __name__ == '__main__':
	start = time.time()
	queue_thread_main()
	print("done: %s" % (time.time() - start))