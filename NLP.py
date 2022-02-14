import os
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from PATHS import ABUSE_PATH, SATOSHI
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist as FD


## Downloads and constants
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


abuse_df = pd.read_csv(ABUSE_CSV_PATH)
STOPWORDS = stopwords.words('english')
#every journy start

# boolean if the string starts with https small_vec.map(lambda x: not x.startswith("https"))
def clean_tokens(list_of_tokens):
    filtered_list = [word for word in list_of_tokens if not (word in STOPWORDS or word in string.punctuation)]
    return filtered_list


def concatSeries(series):
    words_list = [word for list in series for word in list]
    return words_list


def create_all_tokens(series):
    str_series = series.dropna()
    str_series = str_series.str.lower()
    print("Converted to strings..\r")
    token_series = str_series.apply(nltk.word_tokenize)
    print("Tokenizing..\r")
    filtered_token_series = token_series.apply(clean_tokens)
    print("Filetring..\r")
    words = concatSeries(filtered_token_series)
    print("Done.\r")
    return words


def makeTokens():
    print("Making tokens..\r")
    tokens = create_all_tokens(abuse_df.description)
    print("Saving..\r")
    with open(f'/mnt2/abuse_data/tokens.txt', 'w') as f:
        for listitem in tokens:
            f.write('%s\n' % listitem)
        f.close()
    print(f"Saved. Total size {os.path.getsize('/mnt2/abuse_data/tokens.txt')}\r")

