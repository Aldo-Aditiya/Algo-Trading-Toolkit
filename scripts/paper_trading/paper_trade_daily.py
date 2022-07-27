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

''' Import custom Library '''
lib_path = '/workspace/202205_idx-trading/lib'
sys.path.insert(0, lib_path)
# Read Imports
from backtest import Backtest
from utils import read_config
sys.path.remove(lib_path)


''' Read Configs '''
parser = argparse.ArgumentParser(description='')

parser.add_argument('--config_filepath', type=str, help='File path for config file', dest='config_filepath')
parser.add_argument('--active', type=bool, help='Whether the paper trading is still active or not', dest='active')

FLAGS = parser.parse_args()

# Get Relevant Config
config_dict = read_config(FLAGS.config_filepath)
run_params = config_dict['run_params']

strat_dir = run_params['base_strat_dir']
strat_filename = run_params['strat_filename']
strat_filepath = strat_dir + strat_filename
benchmark_filepath = run_params['base_benchmark_dir'] + "paper_trading/" + run_params['benchmark_filename']
run_date_start = config_dict['paper_trade_params']['run_date_start']

ticker_weights = config_dict['strat_params']['ticker_weights']


''' Import the Strategy '''
spec = importlib.util.spec_from_file_location("strategy", strat_filepath)
strat = importlib.util.module_from_spec(spec)
sys.modules["strategy"] = strat
spec.loader.exec_module(strat)


''' Run and Generate Strategy Metrics '''
s = strat.Strategy(FLAGS.config_filepath, mode="paper_trade")
s_df = s.run()

b = Backtest()
result_dict = b.run(s_df, ticker_weights, prep_result_to_df=True)

b.strat_df.to_csv(benchmark_filepath)


''' Update paper_trade.csv with current run '''
csv_filepath = '/workspace/202205_idx-trading/_metadata/paper_trade.csv'
csv_df = pd.read_csv(csv_filepath)

# Define and Combine Relevant dict keys
identifier_dict = {
                       'date_start': [run_date_start], 
                       'date_updated': [datetime.now().strftime("%Y-%m-%d")],
                       'day_count': [(datetime.now() - datetime.fromisoformat(run_date_start)).days],
                       'active': [FLAGS.active],
                       'benchmark_name': [os.path.splitext(run_params['benchmark_filename'])[0]]
                  }

filepath_dict = {
                    'config_filepath': [FLAGS.config_filepath],
                    'benchmark_filepath': [benchmark_filepath]
                }

# TODO - If the backtest have newer columns, might want to add more columns (but keep older ones)
buff_dict = dict(identifier_dict, **result_dict)
row_buff = dict(buff_dict, **filepath_dict)

# Replace the strat row with the same benchmark_name by this row_buff, and save
if csv_df.loc[csv_df['benchmark_name'].isin(row_buff['benchmark_name'])].empty:
    csv_df = pd.concat([csv_df, pd.DataFrame(row_buff)], ignore_index=True, sort=False)
else: 
    index = csv_df.loc[csv_df['benchmark_name'].isin(row_buff['benchmark_name'])].index
    csv_df.iloc[index] = pd.DataFrame(row_buff)
csv_df.to_csv(csv_filepath, index=False)