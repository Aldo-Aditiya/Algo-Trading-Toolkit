import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

'''
Pull daily ticker data from LQ45, based on the given list of stocks.
'''

# Data Directory
data_dir = '/workspace/202205_idx-trading/_data/'
data_name = '20220525_lq45_index'
lq45_index_dir = data_dir + data_name + '.csv'

# Parameters
interval = '1d'
period = 'max'
date_added = '2022-05-25'

# Getting data for LQ45 Index
print("Getting Stock Data")
lq45_index_data = yf.download('^JKLQ45',
                            interval=interval,
                            period=period,
                            threads=True,
                            group_by='ticker')

# Save in CSV
lq45_index_data.to_csv(lq45_index_dir)

# Update data.csv with current run
csv_filepath = '/workspace/202205_idx-trading/_metadata/data.csv'
csv_df = pd.read_csv(csv_filepath)

row_buff = {
                'date_added': [date_added],
                'date_updated': [datetime.now().strftime("%Y-%m-%d")],
                'data_name': [data_name],
                'instrument': ['stock_index'],
                'area': ['indonesia'],
                'interval': [interval],
                'data_date_start': [lq45_index_data.index[0]],
                'data_date_end': [lq45_index_data.index[-1]],
                'ticker_list': ['LQ45'],
                'source': ['yfinance'],
                'data_path': [lq45_index_dir],
                'ticker_list_filepath': ['none']
           }

# Replace the data row with the same data_name by this row_buff, and save
if csv_df.loc[csv_df['data_name'].isin(row_buff['data_name'])].empty:
    csv_df = pd.concat([csv_df, pd.DataFrame(row_buff)], ignore_index=True, sort=False)
else:
    index = csv_df.loc[csv_df['data_name'].isin(row_buff['data_name'])].index
    csv_df.iloc[index] = pd.DataFrame(row_buff)
csv_df.to_csv(csv_filepath, index=False)