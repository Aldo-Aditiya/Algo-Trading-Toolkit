import numpy as np
import pandas as pd

import quantstats as qs
qs.extend_pandas()

import os
import sys
import argparse
import json
import importlib
from datetime import datetime, timedelta

def read_config(config_filepath):
    '''
    Read JSON config file as dictionary
    '''
    with open(config_filepath) as f:
        config_dict = json.load(f)
    return config_dict

# Handle Arguments
parser = argparse.ArgumentParser(description='')

parser.add_argument('--config_filepath', type=str, help='File path for config file', dest='config_filepath')

FLAGS = parser.parse_args()

# Import the Strategy
## Get Relevant Config
config_dict = read_config(FLAGS.config_filepath)
run_params = config_dict['run_params']

strat_dir = run_params['base_strat_dir']
strat_filename = run_params['strat_filename']
strat_filepath = strat_dir + strat_filename
benchmark_filepath = run_params['base_benchmark_dir'] + "backtest/" + run_params['benchmark_filename']

run_date_start = config_dict['backtest_params']['run_date_start']
if config_dict['backtest_params']['run_date_end'] == 'full':
    run_date_end = datetime.now().strftime("%Y-%m-%d")
else:
    run_date_end = config_dict['backtest_params']['run_date_end']

## Import Strategy using Importlib
spec = importlib.util.spec_from_file_location("strategy", strat_filepath)
s = importlib.util.module_from_spec(spec)
sys.modules["strategy"] = s
spec.loader.exec_module(s)

# Run the Strategy
strat = s.Strategy(FLAGS.config_filepath, mode="backtest")
strat.run()
strat.df_proc.to_csv(benchmark_filepath)

# Update backtest.csv with current run
csv_filepath = '/workspace/202205_idx-trading/_metadata/backtest.csv'
csv_df = pd.read_csv(csv_filepath)
s_df = strat.df_proc
s_ret = s_df.set_index('Date')['return']

row_buff = {
               'date_start': [run_date_start], 
               'date_end': [run_date_end],
               'date_run': [datetime.now().strftime("%Y-%m-%d")],
               'day_count': [(datetime.now() - datetime.fromisoformat(run_date_start)).days],
               'benchmark_name': [os.path.splitext(run_params['benchmark_filename'])[0]],
               'Cumulative Return': [s_df.iloc[-1]['cum_return']],
               'CAGR': [qs.stats.cagr(s_ret)],
               'Sharpe': [qs.stats.sharpe(s_ret)],
               'Max DD %': [qs.stats.drawdown_details(s_ret)['max drawdown'].min()],
               'Longest DD': [qs.stats.drawdown_details(s_ret)['days'].max()],
               'config_filepath': [FLAGS.config_filepath],
               'benchmark_filepath': [benchmark_filepath]
           }

# Replace the strat row with the same benchmark_name by this row_buff, and save
if csv_df.loc[csv_df['benchmark_name'].isin(row_buff['benchmark_name'])].empty:
    csv_df = pd.concat([csv_df, pd.DataFrame(row_buff)], ignore_index=True, sort=False)
else: 
    index = csv_df.loc[csv_df['benchmark_name'].isin(row_buff['benchmark_name'])].index
    csv_df.iloc[index] = pd.DataFrame(row_buff)
csv_df.to_csv(csv_filepath, index=False)