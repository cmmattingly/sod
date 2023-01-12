import os
os.chdir(os.environ['PROJECT_DIR'])

import requests
import pandas as pd
import json
from tqdm import tqdm 

from utils.helpers import handle_get_request, df_to_csv

HEADERS = {"X-API-KEY": os.environ['PROPUBLICA_KEY']}
MAX_DOCS = 100

# parse response from statements request
def parse_response(response, member_id):
    if not response.text:
        print(f"member_id={member_id} documents not found")
        return None
    
    json_data = json.loads(response.text)['results']

    return [
        [member_id, doc['title'], doc['date'], doc['url']]
        for doc in json_data
    ]

def handle_request_offset(member_id, max_docs=MAX_DOCS):
    data = []
    for offset in range(0, max_docs, 20): # 20 docs per request
        req_url = f"https://api.propublica.org/congress/v1/members/{member_id}/statements/117.json?offset={offset}"
        res = handle_get_request(req_url, headers=HEADERS)
        data += parse_response(res, member_id)
    return data

def main():
    senate_data = pd.read_csv("data/raw/sen_metadata.csv")
    member_ids = senate_data['member_id'].unique()

    documents = []
    for member_id in tqdm(member_ids):
        data = handle_request_offset(member_id)
        documents += data
    
    df = pd.DataFrame(documents, columns=["member_id", "title", "date", "url"])

    df_to_csv(df, path="data/raw/sen_urls.csv", verbose=True)

if __name__ == "__main__":
    main()