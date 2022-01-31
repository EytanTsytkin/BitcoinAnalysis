import requests
import pandas as pd
from io import StringIO
import os

ABUSE_PATH = '/mnt2/abuse_data/abuse.csv'
API_TOKEN = "tcJqsIyaWmQln8JnjjqmMQ4dOIpxQBW8HfmotA86Kmx08FX2M05FSCPHlWTP"

def getAbuseData():
    print("Requesting..\r")
    res = requests.get(f"https://www.bitcoinabuse.com/api/download/forever?api_token={API_TOKEN}")
    print("Making Dataframe..\r")
    AbuseData = pd.read_csv(StringIO(res.content.decode('utf-8')), error_bad_lines=False)
    print("Saving..\r")
    with open(ABUSE_PATH, 'w') as i:
        AbuseData.to_csv(i)




