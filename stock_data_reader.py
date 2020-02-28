# Stable version

import re
import os
import os.path as path
import tensorflow as tf
import pandas as pd
from glob import glob
from tqdm import tqdm
import itertools
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import threading
import time

from preprocessor import Preprocessor
from db_manager import DBManager
print("dependency loaded")

DEBUG_ROOT_PATH = os.path.expanduser("~/Desktop/wrds_data/seconds_data/")

symbol_list = ['FB','JPM','XOM']
# 'FB','JPM','XOM','GOOG','BAC','AAPL','MSFT','AAPL', 'AMZN','JNJ','INTC','CVX','WFC','V','UNH','HD', 'PFE', 'CSCO', 'T'


class StockDataReader():
    def __init__(self,data_dir,
                coord,
                symbol_list,
                year_range,
                symbol_first,
                data_win_len,
                receptive_field,
                queue_size=500):
        # system initialize
        self.db_manager = DBManager(data_dir)
        self.preprocessor = Preprocessor()

        self.coord = coord
        self.threads = []

        # processing params
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.year_range = year_range
        self.symbol_first = symbol_first
        self.data_win_len = data_win_len
        self.receptive_field = receptive_field

        # queue setup
        self.trans_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.trans_queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.trans = self.trans_queue.enqueue([self.trans_placeholder])
        # for multithreading:
        self.yield_list = itertools.product(self.symbol_list, self.year_range) if self.symbol_first else itertools.product(self.year_range, self.symbol_list)
        
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


    def dequeue_trans(self, num_elements):
        return self.trans_queue.dequeue_many(num_elements)

    def main_thread_queue(self, sess):
        stop = False

        # iterate through all symbol and years
        for iter_order_a, iter_order_b in self.yield_list:
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
            resampled_data_list = [self.preprocessor.to_minute_data(data).get("price") for data in new_data_list]
            data_window = self.preprocessor.sliding_window(resampled_data_list, self.data_win_len)

            if data_window is not None:
                processed_data_window = self.preprocessor.batch_log_transform(data_window)

                # making sure the input into the queue has only one dimension
                assert len(np.array(processed_data_window).shape)==1

                sess.run(self.trans, feed_dict={self.trans_placeholder: processed_data_window})


    def start_threads(self, sess, n_threads=2):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.main_thread_queue, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

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

    session_config = tf.ConfigProto(inter_op_parallelism_threads = 0,
                                    intra_op_parallelism_threads = 0)
    with tf.Session(config=session_config) as sess:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader.start_threads(sess)

        for _ in range(8): #19*6-2
            trans = sess.run(trans_dequeue)
            trans_full_list = np.append(trans_full_list, trans)

        coord.request_stop()
        coord.join(threads)


    trans = trans_full_list.reshape(-1,1)
    np.savetxt(os.path.expanduser('~/Desktop/research/transformation/test/tests.csv'), np.array(trans),delimiter=',')


if __name__ == '__main__':
	start = time.time()
	queue_thread_main()
	print("done: %s" % (time.time() - start))