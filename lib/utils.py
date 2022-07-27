import os
import sys
import json

def read_config(config_filepath):
    '''
    Read JSON config file as dictionary
    '''
    with open(config_filepath) as f:
        config_dict = json.load(f)
    return config_dict

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)