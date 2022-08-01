import pandas as pd
import numpy as np
import math

from datetime import datetime

# Utilities
def handle_nan(df, method='bfill'):
    ## Fill NaN values with the earliest data
    if method == 'bfill':
        return df.fillna(method='bfill', axis=0)
    elif method == 'zerofill':
        return df.fillna(0)
    elif method == 'drop':
        return df.dropna()

def extend_price_df(df):
    '''
    Calculates returns, log_returns, and log_prices to a df with 'price' column
    '''
    df['log-price'] = np.log(df['price'])
    df['return'] = df['price'].pct_change()
    df['log-return'] = np.log(1 + df['return'])
    
    df = handle_nan(df, method='zerofill')
    
    return df
     
def gen_combined_df(df_dict, dict_keys, col, nan_handle_method='bfill', add_pfix=True):
    for i, key in enumerate(dict_keys):
        if i == 0:
            df_buff = pd.DataFrame(index=df_dict[key].index)
        for c in col:
            if add_pfix:
                df_buff[c + '_' + key] = df_dict[key][c]
            else:
                df_buff[key] = df_dict[key][c]
    
    # Handle NaN values from combination of multiple tickers
    # Assumes that NaN values because "stock have not existed" has been handled
    df_buff = handle_nan(df_buff, method=nan_handle_method)
    df_buff = handle_nan(df_buff, method='drop')
            
    return df_buff

# Data Generator Classes
class RandomPriceData():
    '''
    Generate random price data based on a selected underlying process
    '''
    # TODO - Add Mean Reverting random data
    
    def __init__(self, random_seed=42):
        # TODO - Might want to add capability to use random_seed to control generated values
        self.random_seed = random_seed
        self.random_series = []
    
    def run_simulation(self, random_f, num_iter=1000, **kwargs):
        # Simulation
        self.simulate(random_f, num_iter=num_iter, **kwargs)
        return self.random_series
    
    # Random Processes
    def random_oscillation(self, n_length=None, t_length=None, amp=None, std_scale=None, mode='sawtooth'):
        '''
        Generate a constant oscillation
        
        "mode" determines the underlying oscillation process, e.g. sawtooth or sine
        '''
        std = amp * std_scale
        
        # Generate Underlying Trend
        if mode=="sawtooth":
            trend_step = amp / t_length
            cycles = np.ceil(n_length/t_length)

            uptrend = list(np.arange(start=-amp/2, stop=amp/2, step=trend_step))
            downtrend = list(np.arange(start=amp/2, stop=-amp/2, step=-trend_step))

            alltrends = [uptrend + downtrend] * int(np.ceil(cycles))
            alltrends = sum(alltrends, [])
            underlying_arr = np.array(alltrends[:n_length])
        
        elif mode=="sine":
            cycles = (np.ceil(n_length/t_length))
            cycles_as_pi = cycles*np.pi
            increment = cycles_as_pi/n_length

            alltrends = [np.sin(x)*amp/2 for x in np.arange(0.0, cycles_as_pi, increment)]
            underlying_arr = np.array(alltrends[:n_length])
        
        # Generate (Normal) Noise
        noise_arr = self.random_gaussian(n_length=n_length, mean=0, std=std)
        
        # Add Underlying and Noise to generate random data
        random_arr = [u + n for (u, n) in zip(underlying_arr, noise_arr)]
        
        # DF Conversion
        ## Generate an Arbitrary Time Series
        date_indices = self.gen_arbitrary_date_indices(len(random_arr))
        ## Convert to DF
        random_series = pd.Series(random_arr, index=date_indices)
        
        return random_series
    
    def random_trend(self, movement="up", frac_change=0.1, frac_change_error=0.05, random_f=None, n_break=100, **kwargs):
        '''
        Generate a random up/down trend series with around frac_change difference in start vs end prices.
        '''
        assert random_f is not None, "random_f must be given"
        
        # Generate Boundaries
        if movement=="up":
            lower_bound = frac_change - frac_change_error
            upper_bound = frac_change + frac_change_error
            
        elif movement=="down":
            lower_bound = -(frac_change + frac_change_error)
            upper_bound = -(frac_change - frac_change_error)
        
        # Generate random series until the boundaries are satisfied
        change_satistfied = False
        n = 0
        while not(change_satisfied) and n < n_break:
            random_series = random_f(**kwargs)
            change = random_series[-1] - random_series[0] / random_series[0]
            
            if lower_bound <= change <= upper_bound:
                change_satisfied = True
                
            n += 1
        
        return random_series
    
    def random_gaussian(self, n_length=None, mean=None, std=None):
        return np.random.normal(loc=mean, scale=std, size=n_length)
            
    def brownian_motion(self, n_length=None, std=None, mode="price"):
        '''
        Generate a price series using brownian motion.
        
        Note: We calculate the returns as brownian motion, and convert it into prices.
        
        Source: http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
        '''
        # Calculate returns as normal
        ## Note: We use a delta-standardized std measure
        ret_arr = self.random_gaussian(n_length=n_length, mean=0, std=std*np.sqrt(1/n_length))
        
        # Return price or returns depending on mode
        date_indices = self.gen_arbitrary_date_indices(len(ret_arr))
        if mode=="price":
            price_series = pd.Series(self.ret_to_prices(ret_arr), index=date_indices)
            return price_series
        elif mode=="logret":
            return pd.Series(ret_arr, index=date_indices)
                      
    def geometric_brownian_motion(self, n_length=None, std=None, drift=None, mode="price"):
        '''
        Generate price series using geometric brownian motion.
        Source: http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
        '''
        # Calculate bm and drift components
        bm_arr = self.brownian_motion(n_length=n_length, std=std, mode="logret").values
        
        drift_arr = (drift - 0.5 * math.pow(std, 2.0)) * 1/n_length
        
        ret_arr = np.add(bm_arr, drift_arr)
        
        # Return price or returns depending on mode
        date_indices = self.gen_arbitrary_date_indices(len(ret_arr))
        if mode=="price":
            price_series = pd.Series(self.ret_to_prices(ret_arr), index=date_indices)
            return price_series
        elif mode=="logret":
            return pd.Series(ret_arr, index=date_indices)
                      
    def jump_diffusion(self, n_length=None, std=None, drift=None, jump_mean=None, jump_std=None, poisson_rate=None, mode="price"):
        '''
        Generate price series using Merton Jump-Diffusion Model
        Source: http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
        '''
        
        # Calculate Jump Diffusion
        s_n = time = 0
        small_lamda = -(1.0 / poisson_rate)
        
        jump_sizes = [0.0 for i in range(0, n_length)]
            
        while s_n < n_length:
            s_n += small_lamda * math.log(np.random.uniform(0, 1))
            for j in range(0, n_length):
                if time * 1/n_length <= s_n * 1/n_length <= (j + 1) * 1/n_length:
                    jump_sizes[j] += self.random_gaussian(n_length=1, mean=jump_mean, std=jump_std)[0]
                    break
            time += 1

        jump_arr = np.array(jump_sizes)
        
        # Calculate Returns
        gbm_arr = self.geometric_brownian_motion(n_length=n_length, std=std, drift=drift, mode="logret").values
        ret_arr = np.add(gbm_arr, jump_arr)
        
        # Return price or returns depending on mode
        date_indices = self.gen_arbitrary_date_indices(len(ret_arr))
        if mode=="price":
            price_series = pd.Series(self.ret_to_prices(ret_arr), index=date_indices)
            return price_series
        elif mode=="logret":
            return pd.Series(ret_arr, index=date_indices)
    
    # Simulate Based on Random Process
    def simulate(self, random_f, num_iter=100, **kwargs):
        '''
        Simulate using a chosen random function random_f, for num_iter times
        '''
        self.random_series = []
        
        for ii in range(num_iter):
            sample_df = random_f(**kwargs)
            self.random_series.append(sample_df)
            
    # Plots
    def plot_random_prices(self):
        '''
        Plot all generated price data from the underlying random process
        '''
        assert self.random_series != [], "No random series has been sampled"
        
        # Aggregate All Data
        random_ret_df = pd.DataFrame(np.array(self.random_series.values).transpose(), 
                                     columns=[str(i) for i in range(len(self.random_series))],
                                     index=self.random_series[0].index)
        
        # Plot each series into plot
        fig = plt.figure(figsize=(30,10))
        plt.plot(random_ret_df, color="lightgrey")
        
        # Plot average into plot
        avg_random = random_ret_df.mean(axis=1)
        plt.plot(avg_random, color="k", linewidth=5.0)
        
        plt.show()
                      
    # Helper Functions
    def ret_to_prices(self, log_returns, starting_price=100):
        '''
        Converts log returns to price
        Source: http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
        '''
        returns = np.exp(log_returns)
        
        # A sequence of prices starting with starting_price
        price_sequence = [starting_price]
        for i in range(1, len(returns)):
            # Add the price at t-1 * return at t
            price_sequence.append(price_sequence[i - 1] * returns[i - 1])
            
        return np.array(price_sequence)
    
    def gen_arbitrary_date_indices(self, n_length, start=datetime(2000,1,1)):
        '''
        Generate arbitrary date indices based on required length and start date
        '''
        return pd.date_range(start=start, periods=n_length)
