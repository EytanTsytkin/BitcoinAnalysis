import requests
import langdetect
import pandas as pd
from PATHS import *
from io import StringIO


def getAbuseData():
    print("Requesting..\r")
    res = requests.get(f"https://www.bitcoinabuse.com/api/download/forever?api_token={API_TOKEN}")
    return res


def lang_detect(row):
    try:
        lang = langdetect.detect(row)
    except (langdetect.lang_detect_exception.LangDetectException, TypeError) as err:
        lang = None
    return lang


def req_to_df(request):
    AbuseData = pd.read_csv(StringIO(request.content.decode('utf-8')), error_bad_lines=False)
    AbuseData.description = AbuseData.description.astype("string")
    AbuseData["language"] = AbuseData.description.apply(lang_detect)
    AbuseData.dropna(subset=["language"], inplace=True)
    with open(ABUSE_PATH, 'w') as i:
        AbuseData.to_csv(i)






