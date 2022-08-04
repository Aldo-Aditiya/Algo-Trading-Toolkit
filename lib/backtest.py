import os
import shutil

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import quantstats as qs
qs.extend_pandas()

import random
import numpy as np
import scipy.stats as ss
from sklearn.utils import resample

from datetime import datetime, timedelta
from tqdm import tqdm
from matplotlib import pyplot as plt

params = {'figure.facecolor': 'w'}
plt.rcParams.update(params)

class Backtest():
    '''
    Standard Walk Forward Backtest
    
    Assumptions:
    - We assume strat_df has standard column names
    - Ticker weight is non dynamic
    - Strategies are binary, and signal follows standard of x_entry and x_close for long and short, or '' for no signal
    - Weighting for each instrument is static and granular (not based on a round number of stocks)
    '''
    
    # TODO - Might want to assert that some ticker-based columns exist (for some functions)
    
    def __init__(self, long_only=False, include_transaction_costs=True, buy_cost_pct=0.05, sell_cost_pct=0.1):
        self.long_only = long_only
        self.include_transaction_costs = include_transaction_costs
        if include_transaction_costs:
            self.buy_cost_pct = buy_cost_pct
            self.sell_cost_pct = sell_cost_pct
            
        self.strat_df = None
        
    def run(self, strat_df, ticker_weights=None, prep_result_to_df=False):
        '''
        Run Backtest and generate results
        '''
        # Setup
        self.init_signal(strat_df, ticker_weights=ticker_weights)
        self.calc_returns()
        self.calc_cum_returns()
        self.remove_unpaired_signals()
        self.calc_hits_and_misses()
        
        # Metrics Calculations
        if prep_result_to_df:
            self.result_dict = self.gen_metrics(elmt_to_list=True)
        else:
            self.result_dict = self.gen_metrics()
        return self.result_dict
    
    def gen_metrics(self, elmt_to_list=False):
        result_dict = {
                            "turnover": self.calc_turnover(),
                            "ratio_of_long": self.calc_ratio_of_longs(),
                            "avg_holding_period": self.calc_avg_holding_period(),

                            "cum_return": self.calc_cum_returns(last=True),
                            "cagr": self.calc_cagr(),
                            "volatility": self.calc_ann_vol(),
                            "skew": self.calc_skew(),
                            "sharpe": self.calc_sharpe(),
                            "prob_sharpe": self.calc_prob_sharpe(),
                            "hit_ratio": self.calc_hit_ratio(),
                            "avg_ret_from_hm": self.calc_avg_returns_from_hits_and_misses(),

                            "max_dd": self.calc_dd(mode="max"),
                            "longest_dd": self.calc_dd(mode="longest"),
                            "currently_dd": self.calc_dd(mode="current")
                       }
        
        if elmt_to_list:
            for key in result_dict:
                result_dict[key] = [result_dict[key]]
        
        return result_dict
    
    def gen_returns_based_metrics(self, s_df=None):
        if s_df is not None:
            result_dict = {
                                "cum_return": self.calc_cum_returns(s_df=s_df, last=True),
                                "cagr": self.calc_cagr(s_df=s_df),
                                "volatility": self.calc_ann_vol(s_df=s_df),
                                "skew": self.calc_skew(s_df=s_df),
                                "sharpe": self.calc_sharpe(s_df=s_df),
                                "prob_sharpe": self.calc_prob_sharpe(s_df=s_df),

                                "max_dd": self.calc_dd(s_df=s_df, mode="max"),
                                "longest_dd": self.calc_dd(s_df=s_df, mode="longest"),
                                "currently_dd": self.calc_dd(s_df=s_df, mode="current")
                          }

            return result_dict
        
        else:
            result_dict = {
                                "cum_return": self.calc_cum_returns(last=True),
                                "cagr": self.calc_cagr(),
                                "volatility": self.calc_ann_vol(),
                                "skew": self.calc_skew(),
                                "sharpe": self.calc_sharpe(),
                                "prob_sharpe": self.calc_prob_sharpe(),

                                "max_dd": self.calc_dd(mode="max"),
                                "longest_dd": self.calc_dd(mode="longest"),
                                "currently_dd": self.calc_dd(mode="current")
                          }

            return result_dict
              
    def init_signal(self, strat_df, ticker_weights=None):
        '''
        Initializes signal dataframe for use in backtesting
        Assumptions: 
        - We assume signal_df is of a particular colnames
        - We assume that signal_df also consists of the close price
        
        Input:
        - signal_df: consists of colnames [price_, signal_, strength_, signal_price]
        
        TODO - Change to support other forms of price sources (e.g. market spread)
        '''
        self.strat_df = strat_df.copy()
        
        self.signal_tickers = self.get_signal_tickers()
        self.tickers = [self.parse_col_str(signal_t)[1] for signal_t in self.signal_tickers]
        
        if ticker_weights != None:
            self.ticker_weights = ticker_weights
        else:
            self.ticker_weights = {}
            for t in self.tickers:
                self.ticker_weights[t] = 1/len(self.tickers)
        
        # Change NaN Values in signals to ""
        for signal_t in self.signal_tickers:
            self.strat_df[signal_t] = self.strat_df[signal_t].fillna("")
        
    def get_strat_df(self):
        return self.strat_df.copy()
    
    def load_strat_df(self, strat_df):
        self.strat_df = strat_df
    
    # Calculate Relevant Metrics
    
    # TODO - Might wanna add capability to calculate PnL based on position
    # TODO - Might wanna add capability to calculate metrics regardless of the class variables
    
    ## Running Metrics
    def calc_positions(self):
        '''
        Calculates position per point in time for each ticker and adds it to a column in strat_df
        '''
        def get_spike(x):
            if x == "long_entry" or x == "short_close":
                return 1
            elif x == "long_close" or x == "short_entry":
                return -1
            else:
                return 0
        
        for t in self.tickers:
            position_t = "position_" + t
            signal_t = "signal_" + t
            
            spike_df = self.strat_df[signal_t].apply(get_spike)
            
            self.strat_df[position_t] = spike_df.cumsum() * self.ticker_weights[t]
    
    def calc_returns(self, s_df=None):
        '''
        Calculates returns from price
        Assumptions: 
        - price, not log-price
        
        TODO - Might wanna rethink this implementation if we use positions
        TODO - Might wanna implement using pandas multiindex/hierarchical columns
        TODO - Might wanna implement parallelized pandas processing
        TODO - Might wanna add capability for stop loss
        TODO - Later on, you can implement returns calculation based on different signal price rather than the closing price
        TODO - Later on you can also calculate returns for each ticker
        '''
        if s_df is not None:
            strat_df = s_df.copy()
        else:
            strat_df = self.strat_df.copy()
        
        for t in self.tickers:
            return_t = "return_" + t
            strat_df[return_t] = np.nan
        strat_df['return'] = np.nan    
        
        last_signal = {}
        for ii in range(0, len(strat_df)):
            returns_buff = 0
            for signal_t in self.signal_tickers:
                t = self.parse_col_str(signal_t)[1]
                price_t = 'price_' + t
                return_t = "return_" + t
                
                if not (t in last_signal.keys()):
                    ticker_return = 0
                elif last_signal[t] == 'long_entry':
                    ticker_return = (strat_df[price_t][ii] - strat_df[price_t][ii-1])/strat_df[price_t][ii-1]  
                elif last_signal[t] == 'short_entry':
                    ticker_return = -(strat_df[price_t][ii] - strat_df[price_t][ii-1])/strat_df[price_t][ii-1]
                elif last_signal[t] == 'long_close' or last_signal[t] == 'short_close':
                    ticker_return = 0
                
                # Rescale ticker_return based on weight for the particular ticker
                ticker_return = ticker_return * self.ticker_weights[t]
                
                if not strat_df[signal_t][ii] == "":
                    last_signal[t] = strat_df[signal_t][ii]
            
                    if self.include_transaction_costs:
                        ticker_return += self.calc_transaction_cost(last_signal[t])
                        
                returns_buff += ticker_return
                strat_df[return_t][ii] = ticker_return
                ticker_return = 0

            strat_df['return'][ii] = returns_buff
            
        if s_df is not None:
            return strat_df
        else:
            self.strat_df = strat_df

    def calc_cum_returns(self, s_df=None, last=False):
        '''
        Calculates cumulative return per point in time and adds it to a column in strat_df
        '''
        if s_df is not None:
            strat_df = s_df.copy()
        else:
            strat_df = self.strat_df.copy()
        
        assert "return" in strat_df.columns.values.tolist(), "return is not found in strat_df" 
        strat_df['cum_return'] = (1 + strat_df["return"]).cumprod() - 1
        
        if s_df is not None:
            if last:
                return strat_df['cum_return'][-1]
            else:
                return strat_df
        else:
            self.strat_df = strat_df
            if last:
                return strat_df['cum_return'][-1]
    
    def calc_hits_and_misses(self):
        '''
        Determines if a position is a hit or miss, decided at exit.
        - Hit : Positive Returns
        - Miss: Negative Returns
        '''
        assert "return" in self.strat_df.columns.values.tolist(), "return is not found in strat_df" 
        
        hm_l = []
        cum_return = {}
        last_signal = {}
        
        # Initialize H/M column
        for signal_t in self.signal_tickers:
            t = self.parse_col_str(signal_t)[1]
            hm_t = "H/M_" + t
            self.strat_df[hm_t] = None
        
        for ii in range(0, len(self.strat_df)):
            tickers_dict = {}
            for signal_t in self.signal_tickers:
                t = self.parse_col_str(signal_t)[1]
                return_t = "return_" + t
                hm_t = "H/M_" + t

                if not (t in cum_return.keys()):
                    cum_return[t] = 0
                
                # Determine the return for the current timestamp, and calculate running cumulative return
                if not (t in last_signal.keys()):
                    ticker_return = 0
                elif last_signal[t] == 'long_entry' or last_signal[t] == 'short_entry':
                    ticker_return = self.strat_df[return_t][ii]
                    cum_return[t] = (1 + cum_return[t]) * (1 + ticker_return) - 1 
                elif last_signal[t] == 'long_close' or last_signal[t] == 'short_close':
                    ticker_return = 0 
                
                # Determine H/M Based on running cumulative return
                if not self.strat_df[signal_t][ii] == "":
                    last_signal[t] = self.strat_df[signal_t][ii]
                    
                    if last_signal[t] == 'long_close' or last_signal[t] == 'short_close':
                        self.strat_df[hm_t][ii] = "hit" if cum_return[t] > 0 else "miss"
                        cum_return[t] = 0
                    else:
                        self.strat_df[hm_t][ii] = "" 
                else:
                    self.strat_df[hm_t][ii] = ""
    
    ## Positions Metrics
    def calc_turnover(self, mode="aggregate"):
        '''
        Calculates turnover, or number of trades
        
        TODO - Might wanna double check if I eventually use stop loss
        '''
        turnover = {}
        for signal_t in self.signal_tickers:
            
            l_entry_trade = len(self.strat_df[self.strat_df[signal_t] == 'long_entry'])
            s_entry_trade = len(self.strat_df[self.strat_df[signal_t] == 'short_entry'])
            l_close_trade = len(self.strat_df[self.strat_df[signal_t] == 'long_close'])
            s_close_trade = len(self.strat_df[self.strat_df[signal_t]== 'short_close'])
        
            l_trade = min([l_entry_trade, l_close_trade])
            s_trade = min([s_entry_trade, s_close_trade])
            
            t = self.parse_col_str(signal_t)[1]
            turnover[t] = l_trade + s_trade
            
        if mode=="detailed":
            return turnover
        elif mode=="aggregate":
            return sum(turnover.values())
    
    def calc_ratio_of_longs(self, mode="aggregate"):
        '''
        Calculates ratio of longs for all trades
        '''
        turnover = {}
        longs = {}
        for signal_t in self.signal_tickers:
            
            l_entry_trade = len(self.strat_df[self.strat_df[signal_t] == 'long_entry'])
            s_entry_trade = len(self.strat_df[self.strat_df[signal_t] == 'short_entry'])
            l_close_trade = len(self.strat_df[self.strat_df[signal_t] == 'long_close'])
            s_close_trade = len(self.strat_df[self.strat_df[signal_t]== 'short_close'])
        
            l_trade = min([l_entry_trade, l_close_trade])
            s_trade = min([s_entry_trade, s_close_trade])
            
            t = self.parse_col_str(signal_t)[1]
            turnover[t] = l_trade + s_trade
            longs[t] = l_trade
            
        if mode=="detailed":
            long_ratio = {}
            for t in turnover.keys:
                long_ratio[t] = longs[t] / turnover[t]
            return long_ratio
        elif mode=="aggregate":
            try:
                long_ratio = sum(list(longs.values())) / sum(list(turnover.values()))
            except:
                long_ratio = 0
            return long_ratio
    
    def calc_avg_holding_period(self, mode="aggregate"):
        '''
        Calculates average holding period for each position (short and long combined)
        
        Assumption:
        - After x_entry will always be x_exit
        - For a ticker, longs and shorts will always come in different periods
        - Only calculates holding period for pair sof entry/exit
        '''
        
        avg_holding_periods = {}
        total_holding_period = 0
        num_period = 0
        
        for signal_t in self.signal_tickers:
            t = self.parse_col_str(signal_t)[1]
            holding_periods = []
            buff_df = self.strat_df[self.strat_df[signal_t] != ""]
            buff_df['diff'] = buff_df.index.to_series().diff().dt.days
            
            buff_df = buff_df[buff_df[signal_t].isin(["long_close", "short_close"])]
            avg_holding_periods[t] = buff_df['diff'].mean()
            
            total_holding_period += buff_df['diff'].sum()
            num_period += len(buff_df['diff'])
        
        if mode=="detailed":
            return avg_holding_periods
        elif mode=="aggregate":
            return total_holding_period / num_period
            
    ## Performance Metrics
    def calc_cagr(self, s_df=None):
        '''
        Calculates CAGR
        
        Assumptions:
        - strat_df is in days
        '''
        if s_df is not None:
            return qs.stats.cagr(s_df['return'])
        else: 
            return qs.stats.cagr(self.strat_df['return'])
    
    def calc_ann_vol(self, s_df=None):
        '''
        Calculates annualized volatility
        
        Assumptions:
        - strat_df is in days
        '''
        if s_df is not None:
            return qs.stats.volatility(s_df['return'])
        else:
            return qs.stats.volatility(self.strat_df['return'])
    
    def calc_skew(self, s_df=None):
        '''
        Calculate the returns skew, assuming normal returns distribution
        '''
        if s_df is not None:
            return ss.skew(s_df['return'])
        else:   
            return ss.skew(self.strat_df['return'])
        
    def calc_sharpe(self, s_df=None):
        '''
        Calculates Sharpe Ratio
        
        Assumptions:
        - strat_df is in days
        '''
        if s_df is not None:
            return qs.stats.sharpe(s_df['return'])
        else:
            return qs.stats.sharpe(self.strat_df['return'])
    
    def calc_prob_sharpe(self, s_df=None, sr_benchmark=0):
        '''
        Calculates probability Sharpe Ratio based on a benchmark Sharpe Ratio
        '''
        if s_df is not None:
            ret = s_df['return']
        else:
            ret = self.strat_df['return']
        
        sr = qs.stats.sharpe(ret)
        n = len(ret)
        skew = ss.skew(ret)
        kurtosis = ss.kurtosis(ret, fisher=False)

        # Assuming SR is annualized, we need to change into periodical
        sr = sr/np.sqrt(252)
        sr_benchmark = sr_benchmark/np.sqrt(252)

        sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))
        psr = ss.norm.cdf((sr - sr_benchmark) / sr_std)

        return psr
    
    def calc_deflated_sharpe(self, sr_std, num_trials):
        '''
        Calculates deflated Sharpe Ratio based on number of strategies trialed, the std of the trialed strategy sharpe ratios.
        
        Inputs:
        - sr_std -> Standard Deviation of the Strategie trials's Sharpes
        - num_trials -> Number of Strategies trialed
        '''
        def expected_sr_max(trials_sr_std=0, num_trials=0, exp_sr_mean=0):
            emc = 0.5772156649
            max_z = (1 - emc) * ss.norm.ppf(1 - 1./num_trials) + emc * ss.norm.ppf(1 - 1./(num_trials * np.e))
            return exp_sr_mean + (trials_sr_std*max_z)
        
        ret = self.strat_df['return']

        exp_sr_max = expected_sr_max(trials_sr_std=sr_std, num_trials=num_trials)
        d_sr = self.calc_prob_sharpe(ret, sr_benchmark=exp_sr_max)
        return d_sr
    
    def calc_hit_ratio(self, mode="aggregate"):
        '''
        Calculate the ratio of bets that results in hit.
        '''
        hms = {}
        total_hits = 0
        total_bets = 0
        
        for signal_t in self.signal_tickers:
            t = self.parse_col_str(signal_t)[1]
            hm_t = "H/M_" + t
            
            hm_df = self.strat_df[hm_t][self.strat_df[hm_t] != ""] 
            
            if len(hm_df) == 0:
                hms[t] = 0
                continue
            
            num_hits = len(hm_df[hm_df == "hit"])
            num_bets = len(hm_df)
            
            total_hits += num_hits
            total_bets += num_bets
            
            hms[t] = num_hits / num_bets
        
        if mode=="detailed":
            return hms
        elif mode=="aggregate":
            if total_bets == 0: 
                return 0
            else:
                return total_hits / total_bets
    
    def calc_avg_returns_from_hits_and_misses(self, mode="aggregate"):
        '''
        Calculate the average returns from both hits and misses
        '''
        def calc_return_slice(df, signal_t, price_t):
            if df[signal_t][0] == 'long_entry':
                df['return'] = df[price_t].pct_change()
                
            elif df[signal_t][0] == 'short_entry':
                df['return'] = -df[price_t].pct_change()
        
        hm_avgs = {}
        total_hit_return = 0
        total_hits = 0
        total_miss_return = 0
        total_miss = 0
        
        for signal_t in self.signal_tickers:
            t = self.parse_col_str(signal_t)[1]
            hm_t = "H/M_" + t
            price_t = price_t = 'price_' + t
            
            # Get Indexes of Pairs of Entries and Exits, categorize into hits or misses
            buff_df = self.strat_df[self.strat_df[signal_t] != ""]
            
            ## Append indexes of entry/exit pair to list
            nx_hit_pair = []
            nx_miss_pair = []
            for ii, row_obj in enumerate(buff_df.iterrows()):
                row = row_obj[1]
                if ii == 0:
                    pair_idx = []
                pair_idx.append(row_obj[0])
                
                if ii % 2 != 0: 
                    if row[hm_t] == "hit":
                        nx_hit_pair.append(pair_idx)
                    elif row[hm_t] == "miss":
                        nx_miss_pair.append(pair_idx)
                    pair_idx = []
            
            # Calculate Cumulative Returns between each entry/exit index pair
            hit_cum_ret = []
            miss_cum_ret = []
            
            for pair_idx in nx_hit_pair:
                pair_df = self.strat_df[[signal_t, price_t]][pair_idx[0]:pair_idx[1]]
                calc_return_slice(pair_df, signal_t, price_t)
                hit_cum_ret.append(self.calc_cum_returns(s_df=pair_df, last=True))
                
            for pair_idx in nx_miss_pair:
                pair_df = self.strat_df[[signal_t, price_t]][pair_idx[0]:pair_idx[1]]
                calc_return_slice(pair_df, signal_t, price_t)
                miss_cum_ret.append(self.calc_cum_returns(s_df=pair_df, last=True))
            
            # Average for both hits and misses
            total_hit_return += sum(hit_cum_ret)
            total_hits += len(hit_cum_ret)
            
            total_miss_return += sum(miss_cum_ret)
            total_miss += len(miss_cum_ret)
            
            try:
                hit_avg = sum(hit_cum_ret) / len(hit_cum_ret)
            except:
                hit_avg = 0
                
            try:
                miss_avg = sum(miss_cum_ret) / len(miss_cum_ret)
            except:
                miss_avg = 0
            hm_avgs[t] = [hit_avg, miss_avg]
        
        if mode=="detailed":
            return hm_avgs
        elif mode=="aggregate":
            try:
                avg_hit_ret = total_hit_return/total_hits
            except:
                avg_hit_ret = 0
            
            try:
                avg_miss_ret = total_miss_return/total_miss
            except:
                avg_miss_ret = 0
            
            return avg_hit_ret, avg_miss_ret
    
    ## Drawdowns Metrics
    def calc_dd(self, s_df=None, mode="max"):
        '''
        Calculates drawdown metrics
        - Maximum Drawdown (max dd) for the maximum drawdown which happens for the strategy
        - Time underwater (time underwater) is the length of maximum drawdown
        - Longest Drawdown (longest dd) for the logest days of drawdown whcih happens for the strategy
        - Currently Drawdown (currently dd) for checking if recently (wihtin delta days) the strategy is currently in drawdown
        
        Assumptions:
        - strat_df is in days
        
        Input
        - mode : can be either "max" for max dd, "tuw" for time underwater, "long" for longest dd, and "current" for currently dd
        - delta: used to determine the time delta to consider if something is in drawdown or not
        '''
        if s_df is not None:
            strat_df = s_df
        else:
            strat_df = self.strat_df
        
        if mode=="max":
            return qs.stats.max_drawdown(strat_df['return'])
        
        elif mode=="tuw":
            # TODO - Implement This
            pass
        
        elif mode=="longest":
            dd_series = qs.stats.to_drawdown_series(strat_df['return'])
            
            longest = 0
            longest_index = [None, None]
            buff = 0
            buff_index = [None, None]
            
            for ii, dd in dd_series.iteritems():
                if dd == 0:
                    buff_index[1] = ii
                    
                    longest = buff
                    longest_index = buff_index
                    buff = 0
                    buff_index = [None, None]
                    
                else:
                    if buff == 0:
                        buff_index[0] = ii
                    buff += 1
            
            if buff_index[1] is None:
                buff_index[1] = ii
                
                longest = buff
                longest_index = buff_index
            
            return longest
        
        elif mode=="current":
            dd_series = qs.stats.to_drawdown_series(strat_df['return'])

            if dd_series[-1] < 0:
                indicator = True
            else:
                indicator = False

            return indicator
    
    def calc_hhi(self):
        # TODO - Implement HHI
        pass
    
    # Plot Relevant Plots
    # TODO - Add Plot Save Image Function
    # TODO - Capability to Plot only certain time ranges
    def plot_cum_returns(self, strat_df=None):
        '''
        Plots cumulative returns
        
        Assumptions:
        - strat_df is in days
        '''
        if strat_df is not None:
            qs.plots.returns(strat_df['return'])
        else:
            qs.plots.returns(self.strat_df['return'])
    
    def plot_monthly_returns(self, strat_df=None):
        '''
        Plots monthly returns
        
        Assumptions:
        - strat_df is in days
        '''
        if strat_df is not None:
            qs.plots.monthly_heatmap(strat_df['return'])
        else:
            qs.plots.monthly_heatmap(self.strat_df['return'])
    
    def plot_drawdown(self, strat_df=None):
        '''
        Plots drawdown chart
        
        Assumptions:
        - strat_df is in days
        '''
        if strat_df is not None:
            qs.plots.drawdown(strat_df['return'])
        else:
            qs.plots.drawdown(self.strat_df['return'])
    
    def plot_buy_sell(self, strat_df=None, positions_plot=True):
        '''
        Plots buy/sell plot and positions plot
        
        Assumptions:
        - That 'positions' has been calculated
        '''
        if strat_df is not None:
            buff_s_df = strat_df
        else:
            buff_s_df = self.strat_df
            
        signal_tickers = self.get_signal_tickers(strat_df=buff_s_df)
            
        for signal_t in signal_tickers:
            fig = plt.figure(figsize=(30,10))
            
            t = self.parse_col_str(signal_t)[1]
            price_t = "price_" + t
            position_t = "position_" + t
            
            # Buy/Sell Plot
            ax = plt.subplot(4, 1, (1,3))
            ax.set_title(f"Buy/Sell Plot for {t}")

            ## Plot Price Graph
            ax.plot(buff_s_df[price_t])

            ## Plot Buy and Sell Dots
            for ii, row in buff_s_df.iterrows():
                if row[signal_t] != '':
                    if row[signal_t] == 'long_entry':
                        dot_color = 'go'
                    elif row[signal_t] == 'long_close':
                        dot_color = 'ro'
                    elif row[signal_t] == 'short_entry':
                        dot_color = 'bo'
                    elif row[signal_t] == 'short_close':
                        dot_color = 'yo'
                    ax.plot(ii, row[price_t], dot_color)
            
            # Positions Plot
            if positions_plot:
                ax = plt.subplot(4, 1, 4)
                ax.plot(buff_s_df[position_t])
                
            plt.show()
            
    def plot_metrics_distribution(self, results_df=None, metric=None):
        '''
        Plot distributions, be it metric or returns
        '''
        assert metric is not None, "Metric must be chosen according to column within s_df"
        
        fig = plt.figure(figsize=(30,10))
        
        if results_df is not None:
            try:
                results_df[metric].plot.hist(bins=100)
            except:
                raise ValueError("Given metric must exist within the dataframe")
        else:
            try:
                self.strat_df[metric].plot.hist(bins=100)
            except:
                raise ValueError("Given metric must exist within the dataframe")
            
    # Helper Functions
    def get_signal_tickers(self, strat_df=None):
        
        if strat_df is not None:
            buff_s_df = strat_df
        else:
            buff_s_df = self.strat_df
        
        colnames = buff_s_df.columns.values.tolist()
        signal_tickers = [cs for cs in colnames 
                            if self.parse_col_str(cs)[0]=='signal']
        return signal_tickers
    
    def calc_transaction_cost(self, last_signal):
        if last_signal == 'long_entry' or last_signal == 'short_entry':
            return -self.buy_cost_pct / 100
        elif last_signal == 'long_close' or last_signal == 'short_close':
            return -self.sell_cost_pct / 100
    
    def parse_col_str(self, col_str):
        return col_str.split('_')
    
    def remove_unpaired_signals(self):
        '''
        Remove signals without a pair. e.g. x_entry without x_exit, and vice versa.
        '''
        for signal_t in self.signal_tickers:
            buff_s_df = self.strat_df[self.strat_df[signal_t] != ""][signal_t]
            
            if len(buff_s_df) == 0: continue
            
            if buff_s_df[0] == "long_close" or buff_s_df[0] == "short_close":
                buff_index = buff_s_df.index[0]
                self.strat_df[signal_t][buff_index] = ""
            
            if buff_s_df[-1] == "long_entry" or buff_s_df[0] == "short_entry":
                buff_index = buff_s_df.index[-1]
                self.strat_df[signal_t][buff_index] = ""
                
class HistoricalScenarioBacktest(Backtest):
    '''
    Historical Scenario Backtest
    - Like standard Walk-Forward Backtest, but for notable historical market scenarios (e.g. crash of 2008, etc.)
    
    Assumptions
    - Follow the assumptions of Walk-Forward Backtest
    - Relevant signal date period are already given
    
    Note: The class implementation is not necessary, but for now will use this to save the notable market conditions date, 
          which is used to calculate the signals.
    '''
    def __init__(self, long_only=False, include_transaction_costs=True, buy_cost_pct=0.05, sell_cost_pct=0.1):
        super().init(long_only=long_only, include_transaction_costs=include_transaction_costs, buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)
        self.define_notable_scenarios()
        
    def define_notable_scenarios(self):
        '''
        Note: A lot of the market conditions are observations from the price of LQ45 index
        
        Scenario Description:
        - 2006 Pre-GFC Bull Run -> Used to test whether system can detect the overarching positive trend, despite some setbacks
        - 2008 Great Financial Crisis -> Used to test how the system will perform in a multi-year bear market
        - 2009 Post-GFC Bull Run -> Used to test whether system can detect the overarching positive trend, despite some setbacks
        - 2013 May Turbulence -> Used to test how the system will perform in a sudden drop, followed by a dead cat bounce
        - 2015 April Drop -> Used to test how the system will handle a sudden drop
        - 2015-2016 Turbulence -> Used to test how the system will handle an turbulent upwards market
        - 2018 February Drop -> Used to test how the system will handle a sudden drop
        - 2018-2019 Turbulence -> Used to test how the system will handle a sideways market
        - 2020 Covid Scare -> used to test how the system will handle a sudden catasthropic drop
        
        '''
        self.scenarios = {
                            "2006 Pre-GFC Bull Run": ["2005-11-07", "2007-10-09"],
                            "2008 Great Financial Crisis": ["2007-10-09", "2009-03-06"],
                            "2009 Post-GFC Bull Run": ["2009-03-06", "2013-05-20"],
                            "2013 May Turbulence": ["2013-05-13", "2013-12-16"],
                            "2015 April Drop": ["2015-05-30", "2015-09-21"],
                            "2015-2017 Turbulence": ["2015-10-05", "2017-03-06"],
                            "2018 February Drop": ["2018-01-22", "2018-04-30"],
                            "2018-2019 Turbulence": ["2018-04-30", "2020-01-01"],
                            "2020 Covid Scare": ["2020-01-01", "2020-03-17"]
                      }
        
    def get_notable_scenarios_date(self, scenario):
        return self.scenarios[scenario]
    
class RandomizedBacktest(Backtest):
    '''
    Backtest over different randomization or resampling configs of the original data.
    
    Assumptions
    - Follow the assumptions of Walk-Forward Backtest
    - Random configurations cannot be combined with each other
    '''
    def __init__(self, long_only=False, include_transaction_costs=True, buy_cost_pct=0.05, sell_cost_pct=0.1, random_seed=42):
        super().__init__(long_only=long_only, include_transaction_costs=include_transaction_costs, buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)
        
        # TODO - Might want to add capability to use random_seed to control generated values
        self.random_seed = random_seed
        self.random_series = []
    
    def init_signal(self, strat_df, ticker_weights):
        super().init_signal(strat_df, ticker_weights)
        self.get_trade_indexes()
        
    def run_backtest_simulation(self, strat_df, ticker_weights, random_f, num_iter=1000, **kwargs):
        '''
        Run backtest simulation using defined random function
        '''
        # Setup
        self.init_signal(strat_df, ticker_weights)
        self.calc_returns()
        self.calc_cum_returns()
        
        # Simulation
        self.simulate(random_f, num_iter=num_iter, **kwargs)
        
        random_results = []
        if random_f == self.resample_returns or random_f == self.randomize_skip_trades:
            
            # TODO - Might be able to parallelize this
            # Run Simulation and get metrics
            for ii in tqdm(range(num_iter)):
                self.calc_cum_returns(s_df=self.random_series[ii])
                
                result_dict = self.gen_returns_based_metrics(s_df=self.random_series[ii])
                random_results.append(result_dict)
                
            self.random_results_df = pd.DataFrame(random_results)
                
            # Describe the Metrics' Parameters
            random_metrics = []
            for col in self.random_results_df.columns.tolist():
                random_metric_dict = {
                                        "metric": col,
                                        "mean": self.random_results_df[col].mean(),
                                        "std": self.random_results_df[col].std()
                                     }
                random_metrics.append(random_metric_dict)
                
            self.random_metrics = pd.DataFrame(random_metrics) 
        
        elif random_f == self.randomize_historical_data:
            # This function requires one to re-run the signal generation process
            return self.random_series
    
    # Randomization   
    def resample_returns(self, replace=True):
        '''
        Bootstrap returns based on the OOS strategy's average holding period and trading frequency. 
        
        Can be used for:
        - To assess whether the strategy's performance is significant compared to other random strategies with the same 
          returns distribution.
        - To assess the strategy's likely distribution of performance given the distribution of the returns
        
        Source: Lo, Mamaysky and Wang (2000), from E.Chan's Algorithmic Trading book
        '''
        resampled_returns = resample(self.strat_df['return'].values, n_samples = len(self.strat_df), replace=replace)
        resampled_returns_df = pd.DataFrame({"return": resampled_returns}, index=self.strat_df.index)
        
        return resampled_returns_df
    
    def randomize_skip_trades(self, skip_frac=0.1):
        '''
        Randomizing trade skipping, which will show how well the strategy handles conditions where we might not be able to execute trades.
        '''
        num_trades_skipped = round(skip_frac * len(self.strat_df))
        
        skip_returns_t = {}
        for signal_t in self.signal_tickers:
            t = self.parse_col_str(signal_t)[1]
            return_t = "return_" + t
            
            skipped_trades = resample(self.trade_idxs[signal_t], n_samples=num_trades_skipped, replace=False)

            skip_returns = self.strat_df[return_t].copy()

            for trade_idx in skipped_trades:
                skip_returns[trade_idx[0]:trade_idx[1]] = 0

            skip_returns_t[return_t] = skip_returns.values
            
        skip_returns_df = pd.DataFrame(skip_returns_t, index=self.strat_df.index)
        skip_returns_df['return'] = skip_returns_df.sum(axis=1)
                
        return skip_returns_df
    
    def randomize_historical_data(self, prob_change=0.1, max_change_frac=0.05):
        '''
        Randomly changes historical price data, which will show how robust the strategy is to slight changes in price data.
        '''
        def decision(probability):
            return random.random() < probability
        
        # Select only certain columns
        price_tickers = []
        for signal_t in self.signal_tickers:
                t = self.parse_col_str(signal_t)[1]
                price_tickers.append("price_" + t)
        randomized_history_df = self.strat_df.copy()[price_tickers]
        
        # Randomize price
        for ii in range(0, len(randomized_history_df)):
            for signal_t in self.signal_tickers:
                t = self.parse_col_str(signal_t)[1]
                price_t = "price_" + t
                
                if decision(prob_change):
                    boundary = max_change_frac
                    change_frac = random.sample(list(np.linspace(-boundary, boundary, num=5)), 1)[0]
                    
                    randomized_history_df[price_t][ii] *= (1 + change_frac)
        
        return randomized_history_df
    
    def resample_trades(self):
        '''
        Bootstrap trade and holding period order
        - Used to assess if the strategy's generated signal positioning is statistially significant compared to haphazard orderings.
        '''
        # TODO - Will do later on, since this is similar to what is in randomize returns based on distribution, but a bit more complex to make
        pass
    
    def randomize_strategy_params(self):
        # TODO - Might wanna make a separate strategy class for this
        pass 
    
    def randomize_slippage(self):
        # TODO - Implement this once you have a good enough model of slippage
        pass 
    
    # Simulation Based on Randomization
    def simulate(self, random_f, num_iter=100, **kwargs):
        '''
        Simulate using a chosen random function random_f, for num_iter times
        '''
        self.random_series = []
        
        for ii in range(num_iter):
            sample_df = random_f(**kwargs)
            self.random_series.append(sample_df)
    
    # Plots
    def plot_random_cum_returns(self):
        '''
        Plot cumulative returns of all the generated randomized series.
        '''
        assert self.random_series != [], "No random series has been sampled"
        assert 'cum_return' in self.strat_df.columns.values.tolist(), "Cumultive return of original series has not been calculated."
        
        # Calculate Cumulative Returns
        random_ret_series = []
        for ii, series in enumerate(self.random_series):
            random_ret_series.append(self.calc_cum_returns(s_df=series)['cum_return'].values)
        random_ret_df = pd.DataFrame(np.array(random_ret_series).transpose(), 
                                     columns=[str(i) for i in range(len(random_ret_series))],
                                     index=self.strat_df.index)    
            
        # Plot each series into plot
        fig = plt.figure(figsize=(30,10))
        plt.plot(random_ret_df, color="lightgrey")
        
        # Plot average into plot
        avg_random = random_ret_df.mean(axis=1)
        plt.plot(avg_random, color="k", linewidth=5.0)
        
        # Plot original series into plot
        plt.plot(self.strat_df['cum_return'], color="tab:orange", linewidth=5.0)
        
        plt.show()
            
    # Helper Functions
    def get_trade_indexes(self):
        self.trade_idxs = {}
        for signal_t in self.signal_tickers:
            buff_df = self.strat_df[self.strat_df[signal_t] != ""]

            sig_pairs = []
            for ii, row_obj in enumerate(buff_df.iterrows()):
                row = row_obj[1]
                if ii == 0:
                    pair_idx = []
                pair_idx.append(row_obj[0])

                if ii % 2 != 0: 
                    sig_pairs.append(pair_idx)
                    pair_idx = []

            self.trade_idxs[signal_t] = sig_pairs
        
class ModelBacktest(Backtest):
    def __init__(self, long_only=False, include_transaction_costs=True, buy_cost_pct=0.05, sell_cost_pct=0.1, random_seed=42):
        super().init(long_only=long_only, include_transaction_costs=include_transaction_costs, buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)
        self.random_seed = random_seed