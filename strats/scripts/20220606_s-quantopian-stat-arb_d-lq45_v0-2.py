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


''' Strategy Definition '''
class Strategy():
    '''
    Long-Only Bollinger Band Mean Reversion Strategy
    
    1. Calculate the Rolling Price Spread, by first estimating the beta model of the pair.
       We assume the model is of the form Y = beta * X where Y is the second item in the pair.
    2. Calculate the BBands
    3. Generate Signal, based on BBand (long when price >= bbl, exit when price <= bbm)
    
    Data: LQ45 and Its Index
    '''
    def __init__(self, config_filepath, mode="paper_trade"):
        # Data Directory and Sampling
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
        
        # Strategy Parameters
        strat_params = config_dict['strat_params']
        self.pair = strat_params['pair']
        self.half_life = strat_params['half_life']
        self.beta_lookback = strat_params['beta_lookback']
        self.std = strat_params['std']
        
        ## Dynamically calculate maximum lookback
        self.max_lookback = max(self.beta_lookback, round(self.half_life))

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
    
    def prepare_indicators(self, df_dict):
        
        # Take the relevant price series from each pair
        strat_df = gen_combined_df(df_dict, [self.pair[0], self.pair[1]], ['price'], add_pfix=True)
        price_pair = ["price_" + p for p in self.pair]
        
        # Calculate Rolling Price Spread using Beta Model
        S1 = strat_df[price_pair[0]]
        S1.name = 'price'
        S2 = strat_df[price_pair[1]]
        S2.name = 'price'
        
        S1_indep = sm.add_constant(S1)
        result = RollingOLS(S2, S1_indep, window=self.beta_lookback).fit()
        rolling_beta = result.params['price']
    
        strat_df['beta'] = rolling_beta

        ## calculate rolling spread
        strat_df['spread'] = strat_df[price_pair[1]] - strat_df['beta'] * strat_df[price_pair[0]]

        # Generate Technical Indicators (BBand and SMA)
        lookback = round(self.half_life)
        bbands = ta.bbands(strat_df['spread'], length=lookback, std=self.std)

        bbands_upper_cname = 'BBU' + '_' + str(lookback) + '_' + str(self.std) + '.0'
        bbands_lower_cname = 'BBL' + '_' + str(lookback) + '_' + str(self.std) + '.0'
        bbands_mid_cname = 'BBM' + '_' + str(lookback) + '_' + str(self.std) + '.0'

        strat_df['spread_BBU'] = bbands[bbands_upper_cname]
        strat_df['spread_BBL'] = bbands[bbands_lower_cname]
        strat_df['spread_BBM'] = bbands[bbands_mid_cname]
        
        return strat_df
        
    def gen_signals(self, strat_df):
        
        # Signal Rules
        long_signal = lambda price, bbl: (price <= bbl)
        long_close_signal = lambda price, bbm: (price >= bbm)
        short_signal = lambda price, bbu: (price >= bbu)
        short_close_signal = lambda price, bbm: (price <= bbm)

        # Generate Signals
        last_signal = ''
        strat_df['spread_signal'] = ''
        
        signal_pair = ["signal_" + p for p in self.pair]
        for signal_t in signal_pair:
            strat_df[signal_t] = ''
        
        for i in range(0, len(strat_df)):
            if i == 0:
                strat_df['spread_signal'][i] = ''

            elif last_signal == '':
                if long_signal(strat_df['spread'][i], strat_df['spread_BBL'][i]):
                    strat_df['spread_signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    strat_df[signal_pair[1]][i] = 'long_entry'
                elif long_close_signal(strat_df['spread'][i], strat_df['spread_BBM'][i]):
                    strat_df['spread_signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    strat_df[signal_pair[1]][i] = 'long_close'
                elif short_signal(strat_df['spread'][i], strat_df['spread_BBU'][i]):
                    strat_df['spread_signal'][i] = 'short_entry'
                    last_signal = 'short_entry'
                    strat_df[signal_pair[0]][i] = 'long_entry'
                elif short_close_signal(strat_df['spread'][i], strat_df['spread_BBM'][i]):
                    strat_df['spread_signal'][i] = 'short_close'
                    last_signal = 'short_close'
                    strat_df[signal_pair[0]][i] = 'long_close'
                else:
                    strat_df['spread_signal'][i] = ''

            elif last_signal == 'long_entry':
                if long_close_signal(strat_df['spread'][i], strat_df['spread_BBM'][i]):
                    strat_df['spread_signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    strat_df[signal_pair[1]][i] = 'long_close'
                else:
                    strat_df['spread_signal'][i] = ''

            elif last_signal == 'short_entry':
                if short_close_signal(strat_df['spread'][i], strat_df['spread_BBM'][i]):
                    strat_df['spread_signal'][i] = 'short_close'
                    last_signal = 'short_close'
                    strat_df[signal_pair[0]][i] = 'long_close'
                else:
                    strat_df['spread_signal'][i] = ''

            elif last_signal == 'long_close' or last_signal == 'short_close':
                if long_signal(strat_df['spread'][i], strat_df['spread_BBL'][i]):
                    strat_df['spread_signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    strat_df[signal_pair[1]][i] = 'long_entry'
                elif short_signal(strat_df['spread'][i], strat_df['spread_BBU'][i]):
                    strat_df['spread_signal'][i] = 'short_entry'
                    last_signal = 'short_entry'
                    strat_df[signal_pair[0]][i] = 'long_entry'
                else:
                    strat_df['spread_signal'][i] = ''
                    
        return strat_df
    
    def run(self):
        self.df_data_dict = self.prepare_data()
        self.strat_df = self.prepare_indicators(self.df_data_dict)
        self.strat_df = self.gen_signals(self.strat_df)
        
        return self.strat_df
