# THIS FILE IS FOR IF PROPUBLICA API DOESN'T WORK

import pandas as pd
import json
import re
import argparse

# scraping
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

from utils.helpers import df_to_csv, get_soup

HEADERS = {'User-Agent': 
           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
SENATE_URL = "https://www.senate.gov/senators/index.htm"

# read in state abbreviations
with open('../data/state_abbreviations.txt') as f:
    data = f.read()
    us_state_abbrev = json.loads(data)

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # parse args
    args = parse_args()
    path = args.path

    # get soup object and obtain data from said object
    soup = get_soup(SENATE_URL)
    data = soup.find("table", {"id": "listOfSenators"}).find('tbody').findAll('tr')
    
    # parse data
    sen_metadata = []
    for row in data:
        children = row.findChildren()[1:]
        metadata = [children[0].text, children[0].get('href'), children[1].text, children[2].text, children[4].text]
        sen_metadata.append(metadata)
    
    # create dataframe from data
    df = pd.DataFrame(sen_metadata, columns=["sen_name", "url", "state", "party", "class"])
    
    # convert dataframe to csv
    df_to_csv(df, path)