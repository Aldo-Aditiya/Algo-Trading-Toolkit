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
data_name = '20220525_lq45'
lq45_dir = data_dir + data_name + '/'
lq45_list_dir = data_dir + '20220525_lq45-list.txt'

# Parameters
interval = '1d'
period = 'max'
date_added = '2022-05-25'

# Prepare Stock Tickers
with open(lq45_list_dir, "r") as f:
    lq45_tickers = f.read().split('\n')

## Prepare tickers for international codes
lq45_tickers_international = [f + '.JK' for f in lq45_tickers]

# Getting data for stocks
print("Getting Stock Data")
lq45_data = yf.download(lq45_tickers_international, 
                   interval=interval,
                   period=period,
                   threads=True,
                   group_by='ticker')

## Save in csv
for ticker in lq45_tickers_international:
    lq45_data[ticker].to_csv(lq45_dir + ticker + '.csv')

    '''
# Getting data for LQ45 Index
lq45_index_data = yf.download('^JKLQ45',
                            interval=interval,
                            period=period,
                            threads=True,
                            group_by='ticker')
#print('Last LQ45 Date : ' + str(lq45_index_data.index[-1]))

# Save in CSV
lq45_index_data.to_csv(data_dir + '20220525_lq45_index' + '.csv')
'''

# Update data.csv with current run
csv_filepath = '/workspace/202205_idx-trading/_metadata/data.csv'
csv_df = pd.read_csv(csv_filepath)

row_buff = {
                'date_added': [date_added],
                'date_updated': [datetime.now().strftime("%Y-%m-%d")],
                'data_name': [data_name],
                'instrument': ['stock'],
                'area': ['indonesia'],
                'interval': [interval],
                'data_date_start': [lq45_data['ADRO.JK'].index[0]],
                'data_date_end': [lq45_data['ADRO.JK'].index[-1]],
                'ticker_list': [lq45_tickers_international],
                'source': ['yfinance'],
                'data_path': [lq45_dir],
                'ticker_list_filepath': [lq45_list_dir]
           }

# Replace the data row with the same data_name by this row_buff, and save
if csv_df.loc[csv_df['data_name'].isin(row_buff['data_name'])].empty:
    csv_df = pd.concat([csv_df, pd.DataFrame(row_buff)], ignore_index=True, sort=False)
else: 
    index = csv_df.loc[csv_df['data_name'].isin(row_buff['data_name'])].index 
    csv_df.iloc[index] = pd.DataFrame(row_buff)
csv_df.to_csv(csv_filepath, index=False)