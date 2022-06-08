import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import pandas_ta as ta
import quantstats as qs
qs.extend_pandas()

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import json
import datetime

class Strategy():
    '''
    Long-Only Bollinger Band Mean Reversion Strategy
    
    1. Calculate the Rolling Price Spread, by first estimating the beta model of the pair.
       We assume the model is of the form Y = beta * X where Y is the second item in the pair.
    2. Calculate the BBands
    3. Generate Signal, based on BBand (long when price >= bbl, exit when price <= bbm)
    
    Data: LQ45 and Its Index
    '''
    def __init__(self, config_filepath):
        # Data Directory and Sampling
        # TODO - Change this according to the config template
        config_dict = read_config(config_filepath)
        self.data_dir = config_dict['base_data_dir']
        self.lq45_dir = config_dict['lq45_dir']
        self.lq45_index_filename = config_dict['lq45_index_filename']
        self.lq45_list_filename = config_dict['lq45_list_filename']
        # TODO - Dynamically calculate out of sample start date
        self.out_sample_date_start = config_dict['out_sample_date_start']
        self.benchmark_filepath = config_dict['base_benchmark_dir'] + config_dict['benchmark_filename']
        
        # Strategy Parameters
        self.pair = config_dict['pair']
        self.half_life = config_dict['half_life']
        self.beta_lookback = config_dict['beta_lookback']
        self.std = config_dict['std']
        
    def read_config(config_filepath):
        with open(config_filepath) as f:
            config_dict = json.load(f)
        return config_dict
    
    def prepare_data(self):
        # Prepare Stock Tickers
        with open(self.data_dir + lq45_list, "r") as f:
            lq45_tickers = f.read().split('\n')

        ## Prepare active tickers for international codes
        active_tickers_international = [f + '.JK' for f in lq45_tickers]

        # Read data from each ticker's csv
        lq45_df_dict = {}
        for ticker in active_tickers_international:
            lq45_df_dict[ticker] = pd.read_csv(self.lq45_dir + ticker + '.csv')
        lq45_index_data = pd.read_csv(self.data_dir + self.lq45_index_filename)

        # Do Some basic data Operations 
        ## (Fill NaN, take only certain data range, generate in sample an dout of sample data)

        lq45_out_df = {}
        for ticker in active_tickers_international:
            ## Fill NaN values with the earliest data
            lq45_df_dict[ticker].fillna(method='bfill', axis=0, inplace=True)

            ## Take In Sample and Out Sample Data
            lq45_df_dict[ticker]['Date'] = pd.to_datetime(lq45_df_dict[ticker]['Date'])
            lq45_out_df[ticker] = lq45_df_dict[ticker][lq45_df_dict[ticker]['Date'] >= self.out_sample_date_start]

            ## Reset Index After Dropped
            lq45_out_df[ticker] = lq45_out_df[ticker].reset_index(drop=True)

        ## Do the same for lq45 index data
        lq45_index_data.fillna(method='bfill', axis=0, inplace=True)
        lq45_index_data['Date'] = pd.to_datetime(lq45_index_data['Date'])
        
        # Combine to one DF Dictionary
        lq45_out_df['LQ45'] = lq45_index_data[lq45_index_data['Date'] >= self.out_sample_date_start]
        lq45_out_df['LQ45'] = lq45_out_df['LQ45'].reset_index(drop=True)
        
        return lq45_out_df
    
    def prepare_indicators(self, df_dict):
        # Take the relevant price series from each pair
        df_proc = pd.concat([df_dict[self.pair[0]]['Date'], df_dict[self.pair[0]]['Adj Close'], df_dict[self.pair[1]]['Adj Close']], 
                                axis=1,
                                keys=['Date', self.pair[0], self.pair[1]])

        # Calculate Rolling Price Spread using Beta Model
        S1 = df_proc[self.pair[0]]
        S2 = df_proc[self.pair[1]]

        S1_indep = sm.add_constant(S1)
        result = RollingOLS(S2, S1_indep, window=self.beta_lookback).fit()
        rolling_beta = result.params[self.pair[0]]

        ## add Rolling Beta to main df
        df_proc['beta'] = rolling_beta

        ## calculate rolling spread
        df_proc['spread'] = df_proc[self.pair[1]] - df_proc['beta'] * df_proc[self.pair[0]]

        # Generate Technical Indicators (BBand and SMA)
        df_proc.set_index('Date')
        lookback = round(self.half_life)
        bbands = ta.bbands(df_proc['spread'], length=lookback, std=self.std)

        bbands_upper_cname = 'BBU' + '_' + str(lookback) + '_' + str(self.std) + '.0'
        bbands_lower_cname = 'BBL' + '_' + str(lookback) + '_' + str(self.std) + '.0'
        bbands_mid_cname = 'BBM' + '_' + str(lookback) + '_' + str(self.std) + '.0'

        df_proc['spread_BBU'] = bbands[bbands_upper_cname]
        df_proc['spread_BBL'] = bbands[bbands_lower_cname]
        df_proc['spread_BBM'] = bbands[bbands_mid_cname]
        
        return df_proc
        
    def gen_signals(self, df_proc):
        # Signal Rules
        long_signal = lambda price, bbl: (price <= bbl)
        long_close_signal = lambda price, bbm: (price >= bbm)

        # Generate Signals
        last_signal = ''
        df_proc['signal'] = ''
        df_proc['signal_ticker'] = ''
        for i in range(0, len(df_proc)):
            if i == 0:
                df_proc['signal'][i] = ''

            elif last_signal == '':
                if long_signal(df_proc['spread'][i], df_proc['spread_BBL'][i]):
                    df_proc['signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    df_proc['signal_ticker'][i] = self.pair[1]
                elif long_close_signal(df_proc['spread'][i], df_proc['spread_BBM'][i]):
                    df_proc['signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    df_proc['signal_ticker'][i] = self.pair[1]
                else:
                    df_proc['signal'][i] = ''

            elif last_signal == 'long_entry':
                if long_close_signal(df_proc['spread'][i], df_proc['spread_BBM'][i]):
                    df_proc['signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    df_proc['signal_ticker'][i] = self.pair[1]
                else:
                    df_proc['signal'][i] = ''

            elif last_signal == 'long_close':
                if long_signal(df_proc['spread'][i], df_proc['spread_BBL'][i]):
                    df_proc['signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    df_proc['signal_ticker'][i] = self.pair[1]
                else:
                    df_proc['signal'][i] = ''
                    
        return df_proc
    
    def calc_returns(self, df_proc):
        '''
        Calculate returns and cumulative returns per entry on dataframe.

        Strategy Returns: 
        - (Short Spread) returns(S2 Buy price, S2 Close Price). 
        - (Long Spread) returns(S1 Buy price, S1 Close Price). 
        '''
        last_signal = ''
        last_ticker = ''
        df_proc['return'] = np.nan
        for i in range(0, len(df_proc)):
            if last_signal == 'long_entry':
                df_proc["return"][i] = (df_proc[last_ticker][i] - df_proc[last_ticker][i-1]) / df_proc[last_ticker][i-1]
            elif last_signal == 'long_close':
                df_proc["return"][i] = 0
            else:
                df_proc["return"][i] = 0

            if not(df["signal"][i] == ''):
                last_signal = df_proc["signal"][i] 
                last_ticker = df_proc['signal_ticker'][i]

        df_proc["cum_return"] = (1 + df_proc["return"]).cumprod()
        return df_proc
    
    def save_results(self, df_proc):
        df_proc.to_csv(self.benchmark_filepath)
    
    def run(self):
        self.df_data_dict = self.prepare_data()
        self.df_proc = self.prepare_indicators(self.df_data_dict)
        self.df_proc = self.gen_signals(self.df_proc)
        self.df_proc = self.calc_returns(self.df_proc)
        self.save_results()
