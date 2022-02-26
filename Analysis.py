import os
import time
import json
import math
import blocksci
import datetime
import numpy as np
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
from PATHS import *

#featuers = pd.read_csv(FEATURE_BOOK_PATH,index_col=["address"],converters={"tags":literal_eval})#
# from ast import literal_eval
# Lets make this one the main feature file!!
# Im commenting out all of the old stuff and moving them down.
# feel free to delete them if you dont use them :)
# most of them are duplicates of stuff in addressBook.py anyway

def string_to_list(string):
    try:
        lst = literal_eval(string)

    except:
        lst = literal_eval(string.replace(", nan",""))
    return lst

def get_feature_book(for_ml=False,fraud=False):
    feature_book = pd.read_csv(FEATURE_BOOK_PATH, index_col=["address"], converters={"tags":string_to_list})
    feature_book.drop("Unnamed: 0",inplace=True,axis=1)
    feature_book.fillna(0, inplace=True)
    feature_book['lifetime'] = feature_book['lifetime'].map(lambda x: x/(60*60*24))
    feature_book['tx_freq_mean'] = feature_book['tx_freq_mean'].map(lambda x: x/(60*60))
    feature_book['total_dollar'] = feature_book['total_dollar'].map(lambda x: np.log(x) if x != 0 else -13)
    feature_book['dollar_spent_per_tx'] = feature_book['dollar_spent_per_tx'].map(lambda x: np.log(x) if x != 0 else -13)
    feature_book['dollar_obtain_per_tx']  =  feature_book['dollar_obtain_per_tx'].map(lambda x : np.log(x) if x != 0 else -13)
    feature_book = feature_book[feature_book.tags.astype(bool)]
    if for_ml:
        wanted_features = ["lifetime", "tx_freq_mean", "tx_freq_std", "tx_type_odds", "consecutive_in_tx_score",
                           "consecutive_out_tx_score", "dollar_obtain_per_tx", "dollar_spent_per_tx", "tx_value_std",
                           "max_fee", "total_num_tx", "total_dollar"]
        if fraud:
             wanted_features.append("is_fraud")
        else:
            wanted_features.append("y")
        return feature_book[wanted_features]
    return feature_book

def chain():
    return blocksci.Blockchain(CONFIG_PATH)

def cc():
    return blocksci.currency.CurrencyConverter(currency='USD',
                                        start=datetime.date(2009,1,3),
                                         end=datetime.date(2021,1,31))

def timeToUnix(datetime):
    return time.mktime(datetime.timetuple())

def BTCtoUSD(btc,time):
    return cc.btc_to_currency(btc, time)

def checkAddress(blockchain,wallet):
    try:
        if blockchain.address_from_string(wallet):
            return True
    except Exception as e:
        print(e)
        return False

################################ analysis_from csv

def exponential(lam,k):
    return math.exp(-1 * lam * k)

def calculate_fee(row,time):
    pass

def num_in_a_row(series):
    flag = True

def extract_features_USD(df):
    """

    """
    features =  activity_density(df) +  symmetry_score(df) + value_statistics(df)
    return features


def activity_density(df):
    """
    Returns a dict with some time related statistics of the wallet
    """
    first_tx = df.time.iloc[0]
    time_vector = np.array(df.time) - first_tx
    if df.shape[0] > 1:
        time_between_txes = np.array([time_vector[idx] -time_vector[idx-1] for idx in range(1,len(time_vector))])
        return [
             time_vector[-1], #"lifetime"
             first_tx, #"first_tx":
             time_between_txes.mean() / 60, #"tx_freq_mean"
             time_between_txes.std() / 60, #"tx_freq_std"
        ]
    else:
        return [
             time_vector[-1], # "lifetime"
             first_tx, # "first_tx"
             None, # "tx_freq_mean":
             None # "tx_freq_std":
        ]


def symmetry_score(df):
    """
    Rrturns a dict with symmetry related attributes.
    in score/ out score tries to capture the notion of symmetry -
    it calculates how often does input txs are seperated by output txs, and vice versa.
    maximum in both is 0.5, minimum is ( (len(df)^2) ** -1 )

        # Reference: https://stackoverflow.com/questions/66441738/pandas-sum-consecutive-rows-satisfying-condition

    """
    tx_types = df.tx_type.value_counts().values
    if len(tx_types) == 1:
        if df.tx_type.iloc[0] == 1:
            # Wallet that only recieves
            tx_type_odds = df.tx_type.value_counts().values[0]
            out_score = df.shape[0]
            in_score = None
        elif df.tx_type.iloc[0] == -1:
            # Wallet that only gives
            tx_type_odds = 0
            out_score = None
            in_score = df.shape[0]
    else:
        df.reset_index(inplace=True)
        tx_type_odds = df.tx_type.value_counts().values[0]/df.tx_type.value_counts().values[1]

        in_condition, out_condition = (df.tx_type - 1).astype(bool), (df.tx_type + 1).astype(bool)
        in_sums, out_sums = (~in_condition).cumsum()[in_condition], (~out_condition).cumsum()[out_condition]

        in_score = (1/df.tx_type.groupby(in_sums).agg(np.sum)).sum()/df.shape[0]
        out_score = (1/df.tx_type.groupby(out_sums).agg(np.sum)).sum()/df.shape[0]

    return [
        tx_type_odds, # 'tx_type_odds' :
        in_score, # consecutive_in_tx_score' :
        out_score #'consecutive_out_tx_score' :
    ]


def value_statistics(df):
    tx_types = df.tx_type.value_counts().values
    if len(tx_types) == 1:
        if df.tx_type.iloc[0] == 1:
            # Wallet that only recieves
            dollar_obtain_per_tx = df.loc[df.tx_type == 1].valueUSD.sum()/df.tx_type.value_counts().values[0]
            dollar_spent_per_tx = 0
            obtain_spent_ratio = None
            wallet_type = -1
        elif df.tx_type.iloc[0] == -1:
            # Wallet that only gives
            dollar_obtain_per_tx = 0
            dollar_spent_per_tx = df.loc[df.tx_type == -1].valueUSD.sum() / df.tx_type.value_counts().values[0]
            obtain_spent_ratio = None
            wallet_type = 1
    else:
        dollar_obtain_per_tx = df.loc[df.tx_type == 1].valueUSD.sum() / df.tx_type.value_counts().values[0]
        dollar_spent_per_tx = df.loc[df.tx_type == -1].valueUSD.sum() / df.tx_type.value_counts().values[1]
        obtain_spent_ratio = dollar_obtain_per_tx/dollar_spent_per_tx
        wallet_type = 0
    return [
        dollar_obtain_per_tx, # 'dollar_obtain_per_tx' :
        dollar_spent_per_tx, # 'dollar_spent_per_tx' :
        obtain_spent_ratio, # 'obtain_spent_ratio' :
        df.valueUSD.std(), # 'tx_value_std' :
        #'tx_value_prob_mean' : None, # this uses the probabilty of having the tx value in its' block
        #'tx_value_prob_std' : None, # this uses the probabilty of having the tx value in its' block
        df.feeUSD.max(),  # 'max_fee' :  most values is 0 so need to think if we want to recalculte or take diff from 0
        # 'fee_prob_mean' :  None, # this uses the probabilty of having the tx fee in its' block
        # 'fee_prob_std' : None, # this uses the probabilty of having the tx fee in its' block
        df.shape[0],  # 'total_num_tx' :
        df.valueUSD.sum(), # total_dollar' :
        wallet_type # wallet_type' :
    ]


# def prob_statistics(df):
#
#     tx_value_prob_mean
#     tx_value_prob_std
#     fee_prob_mean
#     fee_prob_std
#
#     return [
#         tx_value_prob_mean,# 'tx_value_prob_mean' : None, # this uses the probabilty of having the tx value in its' block
#         tx_value_prob_std,# 'tx_value_prob_std' : None, # this uses the probabilty of having the tx value in its' block
#         fee_prob_mean,# 'fee_prob_mean' :  None, # this uses the probabilty of having the tx fee in its' block
#         fee_prob_std# 'fee_prob_std' : None, # this uses the probabilty of having the tx fee in its' block
#     ]


def peers_statistics(df):
    # Feature that counts how many distinct peers a wallet have
    # how many close friends does he have (=peers with more than 2 txs)
    pass


def extract_features_BTC(df):
    odds = df.tx_type.value_counts().values[0]/df.tx_type.value_counts().values[1]
    life_time = df.time.iloc[-1]-df.time.iloc[1]
    activity_density = df.time.std()/60  # std in minutes
    btc_obtain_per_tx = df.loc[df.tx_type == 1].valueBTC.sum()/df.tx_type.value_counts().values[0]
    btc_spent_per_tx = df.loc[df.tx_type == -1].valueBTC.sum()/df.tx_type.value_counts().values[1]
    max_fee = df.feeBTC.max #most values is 0 so need to think if we want to recalculte or take diff from 0
    total_num_tx = df.shape[0]
    total_btc = df.valueBTC.sum()


def heat_cor_view(big_df,wanted_method : str):
    # i referred the input as pandas but this is still raw function
    df_spear_corr = big_df.corr(method=wanted_method)
    im = plt.imshow(df_spear_corr, cmap=plt.get_cmap('bwr'))
    plt.xticks(np.arange(big_df.shape[1]-1), big_df.columns[:-1], rotation=90)
    plt.yticks(np.arange(big_df.shape[1]-1), big_df.columns[:-1])
    plt.tight_layout()
    plt.colorbar()
    plt.show()
    plt.close()


def some_statistics_on_features(big_df,wanted_plot: str):
    # the same as above
    #here just initial numbers we will know better when will have the size of the big df
    nrow = big_df.shape[1]//3
    ncol = 3
    fig, axes = plt.subplots(nrow, ncol,figsize=(20,20))
    axes = axes.flat
    print("hey")
    for i in range(len(axes)):
        print(len(axes))
        big_df.iloc[:,i].plot(kind=wanted_plot,title=big_df.columns[i] ,ax=axes[i],bins=100)
    print("im here")
    plt.tight_layout()
    plt.show()
    plt.close()
    # plt.savefig('/mnt/plots/small_statistics.png')


def extract_sum_txs(block):
    return (BTCtoUSD(sum([tx.input_value for tx in block.txes.to_list()])*SATOSHI,block.time),
            BTCtoUSD(sum([tx.fee for tx in block.txes.to_list()]) * SATOSHI, block.time),
            block.txes.size)



def extract_range(start,end):
    """

    :param start: First block
    :param end: Last Block
    :return: Returns an array: [avg_tx_usd, avg_fee_usd)
    """
    ti = time.time()
    print(f'Reading block no. {start}..',end='\r')
    temp_array =  sum(np.array(chain.map_blocks(extract_sum_txs,start=start,end=end,cpu_count=4)))
    print(f'Done in {time.time()-ti} seconds.')
    return np.array([temp_array[0]/temp_array[2], temp_array[1]/temp_array[2]])


def make_average_tx_and_fee_dict():
    chain = chain()
    cc = cc()
    t = time.time()
    print('Strated extraction..')
    prob_dict = {(1000*k):extract_range((1000*k),((1000*k)+999)) for k in range(0,665)}
    print(f'Done, in {time.time()-t} seconds. Saving..')
    with open('/root/address_book/block_probability_distribution/prob_dict.json','w') as f:
        json.dump(prob_dict,f)
        f.close()
    print('Done. Have a nice day!')





####
# spectral clustring
# herrcial cluster
# mybe k means
#xgboost
#mybe catboost
#desicion tree