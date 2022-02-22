import os
import time
import json
import blocksci
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from PATHS import *

# Lets make this one the main feature file!!
# Im commenting out all of the old stuff and moving them down.
# feel free to delete them if you dont use them :)
# most of them are duplicates of stuff in addressBook.py anyway

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


def calculate_fee(row,time):
    pass

def num_in_a_row(series):
    flag = True

def extract_features_USD(df):
    """

    """
    # Added type Odds to symmetry score. Feel free to use it/work on it
    # type_odds = df.tx_type.value_counts().values[0]/df.tx_type.value_counts().values[1] #values[0] is outputs

    # I changed activity density so it would calculate the avg and std time between txs.
    #life_time = df.time.iloc[-1] - df.time.iloc[1]
    #activity_density = df.time.std()/60  # std in minutes

    # Added all of these guys under value statistics. awesome features love this shit yo
    # dollar_obtain_per_tx = df.loc[df.tx_type == 1].valueUSD.sum()/df.tx_type.value_counts().values[0]
    # dollar_spent_per_tx = df.loc[df.tx_type == -1].valueUSD.sum()/df.tx_type.value_counts().values[1]
    # max_fee = df.feeUSD.max #most values is 0 so need to think if we want to recalculte or take diff from 0
    # total_num_tx = df.shape[0]
    # total_dollar = df.valueUSD.sum()
    features =  symmetry_score(df) + activity_density(df) + value_statistics(df)
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
    im = plt.imshow(df_spear_corr, cmap=plt.get_cmap('coolwarm'))
    plt.xticks(np.arange(big_df.shape[1]), big_df.columns, rotation=90)
    plt.yticks(np.arange(big_df.shape[1]), big_df.columns_)
    plt.colorbar()
    plt.show()
    plt.close()

def some_statistics_on_features(big_df,wanted_plot: str):
    # the same as above
    #here just initial numbers we will know better when will have the size of the big df
    nrow = big_df.shape[1]//3
    ncol = 3
    fig, axes = plt.subplots(nrow, ncol)
    axes = axes.flat
    for i in range(big_df.shape[1]):
        big_df.iloc[:,i].plot(kind=wanted_plot, ax=axes[i])



