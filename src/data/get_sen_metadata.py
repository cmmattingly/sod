import os
import requests
import pandas as pd
import json
from tqdm import tqdm 

from helpers import handle_get_request, df_to_csv
os.chdir(os.environ['PROJECT_DIRECTORY'])

HEADERS = {"X-API-KEY": os.environ['PROPUBLICA_KEY']}
MEMBERS_ENDPOINT = "https://api.propublica.org/congress/v1/117/senate/members.json"

def main():
    # request members endpoint w. ProPublica api
    res = handle_get_request(MEMBERS_ENDPOINT, HEADERS, verbose=True)
    members_data = json.loads(res.text)['results'][0]['members']

    # get specific data where member is in office
    senate_data = [
        [member['id'], f"{member['first_name']} {member['last_name']}", member['state'],  
            member['party'], member['url'], member['senate_class']]
        for member in members_data
        if member['in_office']
    ]
    # convert data to dataframe
    df = pd.DataFrame(senate_data, columns=["member_id", "name", "state", "party", "url", "class"])

    df_to_csv(df, path="data/raw/sen_metadata.csv", verbose=True)


if __name__ == "__main__":
    main()