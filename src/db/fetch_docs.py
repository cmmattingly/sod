import os
os.chdir(os.environ['PROJECT_DIR'])

import pandas as pd
import json
import requests
from tqdm import tqdm 
import argparse

from helpers import handle_get_request, df_to_csv

def process_response(response):
    # convert request text to dictionary
    json_data = json.loads(response.text)
    
    # obtain elements of each json object and parse into 2d array
    parsed_data = [
        [o['doc_id'], o['sen_id'], o['doc_title'], o['doc_link'], o['doc_text']] 
        for o in json_data
    ]

    # [TODO-2] process into member_id, title, date, url, text
    
    # convert data to dataframe
    df = pd.DataFrame(parsed_data, columns=['doc_id', 'sen_id', 'title', 'url', 'text'])
    return df


def main():
    # fetch from docs endpoint and convert to json
    endpoint = "http://das-lab.org:1702/docs"
    
    # get request and process response
    res = handle_get_request(endpoint, verbose=True)
    df = process_response(res)

    # convert dataframe to csv
    df_to_csv(df, path="data/raw/sen_statements.csv", verbose=True)


if __name__ == "__main__":
    main()