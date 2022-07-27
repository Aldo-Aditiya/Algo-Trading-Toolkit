import sys

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import pandas_ta as ta

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import json
from datetime import datetime, timedelta

''' Import custom Library '''
lib_path = '/workspace/202205_idx-trading/lib'
sys.path.insert(0, lib_path)
# Read Imports
from utils import read_config
from data_utils import gen_combined_df, extend_price_df, handle_nan
sys.path.remove(lib_path)


''' Strategy Definition '''
class HedgeRatioKFLinReg():
    # Source: https://www.quantstart.com/articles/kalman-filter-based-pairs-trading-strategy-in-qstrader/
    def __init__(self):
        # Mean of System State, or Beta/Hedge Ratio
        self.theta = np.zeros(2)
        
        # Covariance Matrix of System State
        self.R = None
        
        # Covariance Matrix of System State Noise
        self.delta = 1e-4
        self.wt = self.delta / (1 - self.delta) * np.eye(2)
        
        # Covariance Matrix of Measurement Noise
        self.vt = 1e-3
        
    def update(self, s1_price, s2_price):
        # Create the observation matrix of the latest prices
        # of TLT and the intercept value (1.0) as well as the
        # scalar value of the latest price from IEI
        F = np.asarray([s1_price, 1.0]).reshape((1, 2))
        y = s2_price

        # The prior value of the states \theta_t is
        # distributed as a multivariate Gaussian with
        # mean a_t and variance-covariance R_t
        if self.R is not None:
            self.R = self.C + self.wt
        else:
            self.R = np.zeros((2, 2))

        # Calculate the Kalman Filter update
        # ----------------------------------
        # Calculate prediction of new observation
        # as well as forecast error of that prediction
        yhat = F.dot(self.theta)
        et = y - yhat

        # Q_t is the variance of the prediction of
        # observations and hence \sqrt{Q_t} is the
        # standard deviation of the predictions
        Qt = F.dot(self.R).dot(F.T) + self.vt
        sqrt_Qt = np.sqrt(Qt)

        # The posterior value of the states \theta_t is
        # distributed as a multivariate Gaussian with mean
        # m_t and variance-covariance C_t
        At = self.R.dot(F.T) / Qt
        self.theta = self.theta + At.flatten() * et
        self.C = self.R - At * F.dot(self.R)
        
        return self.theta, sqrt_Qt, et

class Strategy():
    '''
    Kalman Filter Long-Only Bollinger Band Mean Reversion Strategy
    
    1. Calculate the beta/hedge ratio, std, and price spread from KF.
       We assume the model is of the form Y = beta * X where Y is the second item in the pair.
    2. Generate Signal, based on KF std (long when price spread < -std, exit when price spread >= -std)
    3. Calculate Returns
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
            if self.mode == "paper_trade":
                out_df[ticker] = df_dict[ticker][df_dict[ticker].index >= self.run_date_start]

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
    
    def prepare_indicators(self, df_dict, burn_in=4):
        # Take the relevant price series from each pair
        strat_df = gen_combined_df(df_dict, [self.pair[0], self.pair[1]], ['price'], add_pfix=True)
        price_pair = ["price_" + p for p in self.pair]

        # Calculate beta, std, price spread using KalmanFilter
        kf = HedgeRatioKFLinReg()
        S1 = strat_df[price_pair[0]]
        S2 = strat_df[price_pair[1]]
        kf_l = []

        for p1, p2 in zip(S1,S2):
            _, state_std, spread = kf.update(p1, p2)
            kf_l.append({
                            'spread_std': state_std[0][0],
                            'spread': spread[0]
                        })
        
        ## Combine df
        kf_df = pd.DataFrame(kf_l, index=strat_df.index)
        for col in kf_df:
            strat_df[col] = kf_df[col].values

        # Drop burn in periods
        strat_df = strat_df[burn_in:]
        
        return strat_df
        
    def gen_signals(self, strat_df):
        
        # Signal Rules
        long_signal = lambda price, std: (price < -std)
        long_close_signal = lambda price, std: (price >= -std)
        short_signal = lambda price, std: (price > std)  
        short_close_signal = lambda price, std: (price <= std)

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
                if long_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    strat_df[signal_pair[1]][i] = 'long_entry'
                elif long_close_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    strat_df[signal_pair[1]][i] = 'long_close'
                elif short_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'short_entry'
                    last_signal = 'short_entry'
                    strat_df[signal_pair[0]][i] = 'long_entry'
                elif short_close_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'short_close'
                    last_signal = 'short_close'
                    strat_df[signal_pair[0]][i] = 'long_close'
                else:
                    strat_df['spread_signal'][i] = ''

            elif last_signal == 'long_entry':
                if long_close_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    strat_df[signal_pair[1]][i] = 'long_close'
                else:
                    strat_df['spread_signal'][i] = ''

            elif last_signal == 'short_entry':
                if short_close_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'short_close'
                    last_signal = 'short_close'
                    strat_df[signal_pair[0]][i] = 'long_close'
                else:
                    strat_df['spread_signal'][i] = ''

            elif last_signal == 'long_close' or last_signal == 'short_close':
                if long_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    strat_df[signal_pair[1]][i] = 'long_entry'
                elif short_signal(strat_df['spread'][i], strat_df['spread_std'][i]):
                    strat_df['spread_signal'][i] = 'short_entry'
                    last_signal = 'short_entry'
                    strat_df[signal_pair[0]][i] = 'long_entry'
                else:
                    strat_df['spread_signal'][i] = ''
        
        for signal_t in signal_pair:
            strat_df.loc[strat_df.index < self.run_date_start, signal_t] = ""
        strat_df.loc[strat_df.index < self.run_date_start, 'spread_signal'] = ""

        return strat_df
    
    def run(self):
        self.df_data_dict = self.prepare_data()
        self.strat_df = self.prepare_indicators(self.df_data_dict)
        self.strat_df = self.gen_signals(self.strat_df)
        
        return self.strat_df
