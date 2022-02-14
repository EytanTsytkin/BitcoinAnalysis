import os
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
from PATHS import *
import json
from nltk.stem import PorterStemmer



## Downloads and constants
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

frequency_dict = defaultdict(int)
abuse_df = pd.read_csv(ABUSE_PATH)
STOPWORDS = stopwords.words('english')
#every journy start


def select_language_with_iso639_1(lang,abuse_data):
    if len(lang) != 2:
        print("iso 369_1 language code needed")
    elif lang not in abuse_data.language.unique():
        print("no description in that language")
    else:
        return abuse_data.loc[abuse_data.language == lang].copy()


def clean_tokens(list_of_tokens):
    filtered_list = []
    for word in list_of_tokens:
        if not (word in STOPWORDS or word in string.punctuation):
            filtered_list.append(word)
            frequency_dict[word]+=1
    return filtered_list


def remove_rare_words(list_of_tokens,n):
    list_of_tokens = [word for word in list_of_tokens if frequency_dict[word] > n]
    return list_of_tokens


def create_tokens(series):
    str_series = series.astype("string")
    str_series = str_series.str.lower()
    print("Converted to strings..\r")
    token_series = str_series.apply(nltk.word_tokenize)
    print("Tokenizing..\r")
    filtered_token_series = token_series.apply(clean_tokens)
    print("Filetring..\r")
    return filtered_token_series
# the name is english_token

# optional main
# english_df = select_language_with_iso639_1("en",abuse_df)
# corpus = create_tokens(english_df.description)
# english_df = english_df.apply(remove_rare_words)


def save_processed_corups_and_freq_dict_in_english():
    print("Making tokens..\r")
    corpus_series = create_tokens(select_language_with_iso639_1("en", abuse_df))
    print("Saving..\r")
    with open(ABUSE_CORPUS_PATH, 'a') as i:
        corpus_series.to_csv(i)
        i.close()
    with open(FREQ_DICT_EN,'w') as i:
        json.dump(frequency_dict,i)
    print(f"Saved. Total size {os.path.getsize(ABUSE_CORPUS_PATH)}\r")

