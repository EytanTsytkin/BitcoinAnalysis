import os
import spacy
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
from PATHS import *
from gensim import corpora, similarities, models, utils
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
import json
from ast import literal_eval
#texts = pd.read_csv("/root/abuse_data/texts.csv",usecols=["description"],converters={"description":literal_eval})
# id2word = corpora.Dictionary.load("/root/abuse_data/id2word.dict")
from nltk.stem import PorterStemmer



## Downloads and constants
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm",disable=['parser', 'ner'])
frequency_dict = defaultdict(int)
abuse_df = pd.read_csv(ABUSE_PATH)
abuse_df.description = abuse_df.description.astype("string")

#every journy start


def select_language_with_iso639_1(lang,abuse_data):
    if len(lang) != 2:
        print("iso 369_1 language code needed")
    elif lang not in abuse_data.language.unique():
        print("no description in that language")
    else:
        return abuse_data.loc[abuse_data.language == lang].copy()


def clean_tokens(list_of_tokens):
    stpwords = stopwords.words('english')
    filtered_list = []
    for word in list_of_tokens:
        if not (word in stpwords or word in string.punctuation):
            filtered_list.append(word)
            frequency_dict[word]+=1
    return filtered_list


def bigrams(words, bi_min=3, tre_min=10):
    bigram = models.Phrases(words, min_count=bi_min,threshold=tre_min,connector_words=ENGLISH_CONNECTOR_WORDS)
    bigram_mod = models.phrases.FrozenPhrases(bigram)
    return bigram_mod


def lemmatization(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(sent))
    texts_out = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return texts_out


# def remove_rare_words(list_of_tokens,n):
#     list_of_tokens = [word for word in list_of_tokens if frequency_dict[word] > n]
#     return list_of_tokens


def comput_bigram_mod(sentence,bigram_model):
    return bigram_model[sentence]


def create_corpus(series):
    str_series = series.str.lower()
    print("Tokenizing..\r")
    token_series = str_series.apply(utils.simple_preprocess,deacc=True)
    print("Filetring..\r")
    filtered_token_series = token_series.apply(clean_tokens)
    print("create bigram model & execute..\r")
    bigram_mod = bigrams(filtered_token_series)
    bigram = filtered_token_series.apply(comput_bigram_mod, bigram_model=bigram_mod)
    print("lemmatized the bigrams..\r")
    lemmatized_bigram = bigram.apply(lemmatization)
    print("create corpora.Dictionary..\r")
    id2word = corpora.Dictionary(lemmatized_bigram)
    id2word.filter_extremes(no_below=5, no_above=0.85)
    id2word.compactify()
    print("create corpus..\r")
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram
# the name is english_token

# optional main
# english_df = select_language_with_iso639_1("en",abuse_df)
# corpus = create_tokens(english_df.description)
# english_df = english_df.apply(remove_rare_words)


def save_processed_corups_and_freq_dict_in_english():
    print("Making tokens..\r")
    english_abuse = select_language_with_iso639_1("en", abuse_df)
    corpus = create_corpus(english_abuse.description)
    print("save corpus to disk..\r")
    corpora.MmCorpus.serialize(ABUSE_CORPUS_PATH, corpus)
    print("save freq_dict in english to disk")
    with open(FREQ_DICT_EN,'w') as i:
        json.dump(frequency_dict, i)
#     print(f"Saved. Total size {os.path.getsize(ABUSE_CORPUS_PATH)}\r")
#
#
# def lda_npl_algo(final_sent_tokens):
#     dictionary = corpora.Dictionary(final_sent_tokens)
#     corpus = [dictionary.doc2bow(text) for text in final_sent_tokens]
#     #to save corpus corpora.MmCorpus.serialize('/root/abuse_data/abuse_corpus.mm', corpus)
#     model = models.LdaModel(corpus, id2word=dictionary, num_topics=20)
#     index = similarities.MatrixSimilarity(model[corpus])


def compute_coherence_values(dictionary, corpus, texts, limit=20, start=5, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamulticore.LdaMulticore(
            corpus=corpus,
            num_topics=20,
            id2word=dictionary,
            chunksize=100,
            workers=7, # Num. Processing Cores - 1
            passes=50,
            eval_every = 1,
            per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values