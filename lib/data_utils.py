import pandas as pd
import numpy as np

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