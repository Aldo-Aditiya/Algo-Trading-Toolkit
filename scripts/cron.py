import os
import sys
import argparse
import json
import importlib
from datetime import datetime, timedelta
import pickle as pkl

import logging

def read_config(config_filepath):
    '''
    Read JSON config file as dictionary
    '''
    with open(config_filepath) as f:
        config_dict = json.load(f)
    return config_dict

# Read Config
cron_config_filepath = "cron_config.json"
config_dict = read_config(cron_config_filepath)

data_pull_file_dir = config_dict["data_pull_file_dir"]
data_pull_run = config_dict["data_pull_run"]

paper_trading_file_dir = config_dict["paper_trading_file_dir"]
paper_trading_config_dir = config_dict["paper_trading_config_dir"]
paper_trading_config_run = config_dict["paper_trading_config_run"]

log_file_dir = config_dict["log_file_dir"]

logging.basicConfig(filename=log_file_dir, 
level=logging.DEBUG, 
format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

# Read Cron History
cron_history = "cron_history.pkl"

logging.info("Run Cron Job")

# Run Data Pull
print("\n---- Data Pull")
for runfile in data_pull_run:
    print("--Run Data Pull On: " + runfile)
    data_pull_filepath = data_pull_file_dir + runfile
    os.system("python " + data_pull_filepath)
    
    logging.info(f"Run Data Pull on {runfile}")

print("\n---- Paper Trading")
# Run Paper Trading
for confile in paper_trading_config_run:
    print("--Run Paper Trading On: " + confile)
    paper_trading_confpath = paper_trading_config_dir + confile
    paper_trading_conf = read_config(paper_trading_confpath)
    paper_trading_filepath = paper_trading_file_dir + paper_trading_conf['run_params']['strat_run_filename']
    
    os.system("python " + 
              paper_trading_filepath + 
              " --config_filepath=" + paper_trading_confpath + 
              " --active=True")
    
    logging.info(f"Run Paper Trading on {runfile}")