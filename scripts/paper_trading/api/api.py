from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import pandas as pd
import math

import logging

app = FastAPI()

base_signal_dir = "/workspace/202205_idx-trading/_benchmarks/paper_trading/"
log_file_dir = "/workspace/202205_idx-trading/_logs/api.log"

## Setup Logging
logging.basicConfig(filename=log_file_dir, 
level=logging.DEBUG, 
format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

@app.get("/signal_generator/{signal_source}/signal")
def function(signal_source: str):
    '''
    Get the newest signal from signal_source
    '''
    # Read csv of signal file
    csv_signal_filepath = base_signal_dir + signal_source + ".csv" 
    df_csv = pd.read_csv(csv_signal_filepath, index_col=0)
    last_entry = df_csv.iloc[-1]
    
    # Collect information from last generated signal
    signal_exists = not(math.isnan(last_entry['signal']))
    
    date_generated = last_entry['Date']
    signal = last_entry['signal'] if signal_exists else "none"
    signal_ticker = last_entry['signal_ticker'] if signal_exists else "none"
    no_lots = 1
    price = last_entry[signal_ticker] if signal_exists else 0
    
    logging.info(f"signal requested from signal_generator {signal_source}")
    
    # Generate Response
    response = {
                    "date_generated": date_generated,
                    "signal": signal,
                    "signal_ticker": signal_ticker,
                    "no_lots": no_lots,
                    "price": price
                }
    return response