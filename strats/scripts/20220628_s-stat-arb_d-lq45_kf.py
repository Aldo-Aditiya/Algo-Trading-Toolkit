import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import pandas_ta as ta

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import json
from datetime import datetime, timedelta


def gen_combined_df(df_dict, dict_keys, col, nan_handle_method='bfill', add_pfix=True):
    for i, key in enumerate(dict_keys):
        if i == 0:
            df_buff = pd.DataFrame(index=df_dict[key].index)
        for c in col:
            if add_pfix:
                df_buff[key + '_' + c] = df_dict[key][c]
            else:
                df_buff[key] = df_dict[key][c]
    
    # Handle NaN values from combination of multiple tickers
    # Assumes that NaN values because "stock have not existed" has been handled
    df_buff = handle_nan(df_buff, method=nan_handle_method)
    df_buff = handle_nan(df_buff, method='drop')
            
    return df_buff

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
        config_dict = self.read_config(config_filepath)
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

    def read_config(self, config_filepath):
        with open(config_filepath) as f:
            config_dict = json.load(f)
        return config_dict
    
    def prepare_data(self):
        # Prepare Stock Tickers
        with open(self.data_dir + self.lq45_list_filename, "r") as f:
            lq45_tickers = f.read().split('\n')

        ## Prepare active tickers for international codes
        active_tickers_international = [f + '.JK' for f in lq45_tickers]

        # Read data from each ticker's csv
        lq45_df_dict = {}
        for ticker in active_tickers_international:
            lq45_df_dict[ticker] = pd.read_csv(self.lq45_dir + ticker + '.csv')
        lq45_index_data = pd.read_csv(self.data_dir + self.lq45_index_filename)

        # Do Some basic data Operations 
        ## (Fill NaN, take only certain data range, generate in sample and out of sample data)

        lq45_out_df = {}
        for ticker in active_tickers_international:
            ## Fill NaN values with the earliest data
            lq45_df_dict[ticker].fillna(method='bfill', axis=0, inplace=True)

            ## Take Out Sample Data
            ## Note: We take data (1) within previous 30 days from run_date_start, and (2) After run_date_start until today.
            lq45_df_dict[ticker]['Date'] = pd.to_datetime(lq45_df_dict[ticker]['Date'])
            if self.mode == "paper_trade":
                buff1_df = lq45_df_dict[ticker][lq45_df_dict[ticker]['Date'] < self.run_date_start].tail(self.max_lookback)
                buff2_df = lq45_df_dict[ticker][lq45_df_dict[ticker]['Date'] >= self.run_date_start]

                lq45_out_df[ticker] = pd.concat([buff1_df, buff2_df], ignore_index=True, sort=False)
            elif self.mode == "backtest":
                lq45_out_df[ticker] = lq45_df_dict[ticker][(lq45_df_dict[ticker]['Date'] >= self.run_date_start) & 
                                                           (lq45_df_dict[ticker]['Date'] <= self.run_date_end)]

            ## Reset Index After Dropped
            lq45_out_df[ticker] = lq45_out_df[ticker].reset_index(drop=True)

        # Do the same for lq45 index data
        lq45_index_data.fillna(method='bfill', axis=0, inplace=True)
        
        lq45_index_data['Date'] = pd.to_datetime(lq45_index_data['Date'])
        buff1_df = lq45_index_data[lq45_index_data['Date'] < self.run_date_start].tail(self.max_lookback)
        buff2_df = lq45_index_data[lq45_index_data['Date'] >= self.run_date_start]
        
        ## Combine to one DF Dictionary
        lq45_out_df['LQ45'] = pd.concat([buff1_df, buff2_df], ignore_index=True, sort=False)
        lq45_out_df['LQ45'] = lq45_out_df['LQ45'].reset_index(drop=True)
        
        return lq45_out_df
    
    def prepare_indicators(self, df_dict, burn_in=4):
        # Take the relevant price series from each pair
        df_proc = pd.concat([df_dict[self.pair[0]]['Date'], df_dict[self.pair[0]]['Adj Close'], df_dict[self.pair[1]]['Adj Close']], 
                                axis=1,
                                keys=['Date', self.pair[0], self.pair[1]])

        # Calculate beta, std, price spread using KalmanFilter
        kf = HedgeRatioKFLinReg()
        S1 = df_proc[self.pair[0]]
        S2 = df_proc[self.pair[1]]
        kf_l = []

        for p1, p2 in zip(S1,S2):
            _, state_std, spread = kf.update(p1, p2)
            kf_l.append({
                            'std': state_std[0][0],
                            'spread': spread[0]
                        })
        
        ## Combine df
        kf_df = pd.DataFrame(kf_l)
        for col in kf_df:
            df_proc[col] = kf_df[col].values

        # Drop burn in periods
        df_proc = df_proc[burn_in:]
        df_proc.reset_index(inplace=True)
        
        return df_proc
        
    def gen_signals(self, df_proc):
        
        # Signal Rules
        long_signal = lambda price, std: (price < -std)
        long_close_signal = lambda price, std: (price >= -std)
        short_signal = lambda price, std: (price > std)  
        short_close_signal = lambda price, std: (price <= std)

        # Generate Signals
        last_signal = ''
        df_proc['signal'] = ''
        df_proc['signal_ticker'] = ''
        pair = self.pair
        
        for i in range(0, len(df_proc)):
            if i == 0:
                df_proc['signal'][i] = ''

            elif last_signal == '':
                if long_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    df_proc['signal_ticker'][i] = pair[1]
                elif long_close_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    df_proc['signal_ticker'][i] = pair[1]
                elif short_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'short_entry'
                    last_signal = 'short_entry'
                    df_proc['signal_ticker'][i] = pair[0]
                elif short_close_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'short_close'
                    last_signal = 'short_close'
                    df_proc['signal_ticker'][i] = pair[0]
                else:
                    df_proc['signal'][i] = ''

            elif last_signal == 'long_entry':
                if long_close_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'long_close'
                    last_signal = 'long_close'
                    df_proc['signal_ticker'][i] = pair[1]
                else:
                    df_proc['signal'][i] = ''

            elif last_signal == 'short_entry':
                if short_close_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'short_close'
                    last_signal = 'short_close'
                    df_proc['signal_ticker'][i] = pair[0]
                else:
                    df_proc['signal'][i] = ''

            elif last_signal == 'long_close' or last_signal == 'short_close':
                if long_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'long_entry'
                    last_signal = 'long_entry'
                    df_proc['signal_ticker'][i] = pair[1]
                elif short_signal(df_proc['spread'][i], df_proc['std'][i]):
                    df_proc['signal'][i] = 'short_entry'
                    last_signal = 'short_entry'
                    df_proc['signal_ticker'][i] = pair[0]
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
            if last_signal == 'long_entry' or last_signal == 'short_entry':
                df_proc["return"][i] = (df_proc[last_ticker][i] - df_proc[last_ticker][i-1]) / df_proc[last_ticker][i-1]
            elif last_signal == 'long_close' or last_signal == 'short_close':
                df_proc["return"][i] = 0
            else:
                df_proc["return"][i] = 0

            if not(df_proc["signal"][i] == ''):
                last_signal = df_proc["signal"][i] 
                last_ticker = df_proc['signal_ticker'][i]

        df_proc["cum_return"] = (1 + df_proc["return"]).cumprod()
        return df_proc
    
    def run(self):
        self.df_data_dict = self.prepare_data()
        self.df_proc = self.prepare_indicators(self.df_data_dict)
        self.df_proc = self.gen_signals(self.df_proc)
        self.df_proc = self.calc_returns(self.df_proc)
