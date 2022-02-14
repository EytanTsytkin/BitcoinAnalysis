import requests
import pandas as pd
from io import StringIO
import langdetect
from PATHS import *
import os


def getAbuseData():
    print("Requesting..\r")
    res = requests.get(f"https://www.bitcoinabuse.com/api/download/forever?api_token={API_TOKEN}")
    print("Making Dataframe..\r")
    AbuseData = pd.read_csv(StringIO(res.content.decode('utf-8')), error_bad_lines=False)
    AbuseData.description = AbuseData.description.astype("string")
    print("Saving..\r")
    with open(ABUSE_PATH, 'w') as i:
        AbuseData.to_csv(i)

def leng_detect(row):
    try:
        lang = langdetect.detect(row)
    except langdetect.lang_detect_exception.LangDetectException:
        lang = None
    return lang


