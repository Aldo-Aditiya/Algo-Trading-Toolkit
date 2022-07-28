import sys

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import pandas_ta as ta

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from datetime import datetime, timedelta


''' Import custom Library '''
lib_path = '/workspace/202205_idx-trading/lib'
sys.path.insert(0, lib_path)
# Read Imports
from utils import read_config
from data_utils import gen_combined_df, extend_price_df, handle_nan
sys.path.remove(lib_path)

class LQ45BaseStrategy():
    '''
    Strategy template for LQ45-based data
    '''
    def __init__(self, config=None, config_filepath=None, mode="paper_trade"):
        # Data Directory and Sampling
        
        if config is not None:
            config_dict = config
        
        elif config_filepath is not None:
            config_dict = read_config(config_filepath)

        run_params = config_dict['run_params']
        
        self.data_dir = run_params['base_data_dir']
        self.lq45_dir = run_params['lq45_dir']
        self.lq45_index_filename = run_params['lq45_index_filename']
        self.lq45_list_filename = run_params['lq45_list_filename']
        
        if mode=="paper_trade":
            self.run_date_start = config_dict['paper_trade_params']['run_date_start']
        elif mode=="backtest":
            self.run_date_start = config_dict['backtest_params']['run_date_start']
            if config_dict['backtest_params']['run_date_end'] == 'full':
                self.run_date_end = datetime.now().strftime("%Y-%m-%d")
            else:
                self.run_date_end = config_dict['backtest_params']['run_date_end']
            
        self.mode = mode

    def prepare_data(self):
        # Prepare Stock Tickers
        with open(self.data_dir + self.lq45_list_filename, "r") as f:
            lq45_tickers = f.read().split('\n')

        ## Prepare active tickers for international codes
        active_tickers = [f + '.JK' for f in lq45_tickers]
        active_tickers.append('LQ45')

        # Do Some basic data Operations 
        nan_handle_method = 'bfill'

        df_dict = {}
        for ticker in active_tickers:
            if ticker == 'LQ45':
                df_dict[ticker] = pd.read_csv(self.data_dir + self.lq45_index_filename)
            else:
                df_dict[ticker] = pd.read_csv(self.lq45_dir + ticker + '.csv')

            ## Take Only Date and Adjusted Close
            df_dict[ticker] = df_dict[ticker][['Date', 'Adj Close']]
            df_dict['Date'] = pd.to_datetime(df_dict[ticker]['Date'])
            df_dict[ticker].set_index(pd.DatetimeIndex(df_dict[ticker]['Date']), inplace=True)

            df_dict[ticker].drop('Date', axis=1, inplace=True)

            ## Convert Adj Close to price
            df_dict[ticker]['price'] = df_dict[ticker]['Adj Close']
            df_dict[ticker].drop('Adj Close', axis=1, inplace=True)

        # Separate into In Sample and Out Sample
        nan_cnt_threshold = 252*2

        in_df = {}
        out_df = {}
        rmv_tickers = []
        for ticker in active_tickers:
            ## Take Out Sample Data
            ## Note: We take data: (1) within previous 30 days from run_date_start, and (2) After run_date_start until today.
            if self.mode == "paper_trade":
                buff1_df = df_dict[ticker][df_dict[ticker].index < self.run_date_start].tail(self.max_lookback)
                buff2_df = df_dict[ticker][df_dict[ticker].index >= self.run_date_start]

                out_df[ticker] = pd.concat([buff1_df, buff2_df], sort=False)

            elif self.mode == "backtest":
                out_df[ticker] = df_dict[ticker][(df_dict[ticker].index >= self.run_date_start) & 
                                                           (df_dict[ticker].index <= self.run_date_end)]

            ## Handle NaN Values
            out_df[ticker] = handle_nan(out_df[ticker], method=nan_handle_method)

            ## Extend price to other values
            out_df[ticker] = extend_price_df(out_df[ticker])

        # Remove tickers that only have small amounts of data
        active_tickers = [t for t in active_tickers if t not in rmv_tickers]
        
        return out_df