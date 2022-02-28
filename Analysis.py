import time
import json
import math
import PATHS
import blocksci
import datetime
import numpy as np
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt

# To Do:
# 1. Incorporate the feature transformations in Update Wallet Vector / Merge
# 2. Probabilistic features are cool! Lets make it happen!
# 3. Peer statistics are also awesome. Maybe we can implement them using blocksci with tx_index.


def get_feature_book(for_ml=False,fraud=False):
    """

    :param for_ml: Drops some columns such as first tx, which might introduce a bias
                   into the training model.
    :param fraud: switches y column of the dataframe to binary classification, rather than
                 multi class.
    :return: pd.Dataframe of the features of the labeled data.
    """
    feature_book = pd.read_csv(PATHS.FEATURE_BOOK_PATH, index_col=["address"], converters={"tags":string_to_list})
    feature_book.drop("Unnamed: 0",inplace=True,axis=1)
    feature_book.fillna(0, inplace=True)

    feature_book['lifetime'] = feature_book['lifetime'].map(lambda x: x/(60*60*24)) # Unix to days
    feature_book['tx_freq_mean'] = feature_book['tx_freq_mean'].map(lambda x: x/(60*60)) # Unix to minutes
    feature_book['total_dollar'] = feature_book['total_dollar'].map(lambda x: np.log(x) if x != 0 else -13) # Log
    feature_book['dollar_spent_per_tx'] = feature_book['dollar_spent_per_tx'].map(lambda x: np.log(x) if x != 0 else -13) # Log
    feature_book['dollar_obtain_per_tx']  =  feature_book['dollar_obtain_per_tx'].map(lambda x : np.log(x) if x != 0 else -13) # Log

    feature_book = feature_book[feature_book.tags.astype(bool)]

    if for_ml:
        wanted_features = ["lifetime",
                           "tx_freq_mean",
                           "tx_freq_std",
                           "tx_type_odds",
                           "consecutive_in_tx_score",
                           "consecutive_out_tx_score",
                           "dollar_obtain_per_tx",
                           "dollar_spent_per_tx",
                           "tx_value_std",
                           "max_fee",
                           "total_num_tx",
                           "total_dollar"]

        if fraud:
             wanted_features.append("is_fraud")

        else:
            wanted_features.append("y")
        return feature_book[wanted_features]

    return feature_book


def string_to_list(string):
    """
    Converts 'tags' column of the features csv from strings of lists to lists.
    """
    try:
        lst = literal_eval(string)
    except:
        lst = literal_eval(string.replace(", nan",""))
    return lst


def chain():
    """
    returns a Blockchain instance
    """
    return blocksci.Blockchain(PATHS.CONFIG_PATH)


def cc():
    """
    returns a Currency converter - BTC to USD
    """
    return blocksci.currency.CurrencyConverter(currency='USD',
                                        start=datetime.date(2009,1,3),
                                         end=datetime.date(2021,1,31))


def BTCtoUSD(btc,time):
    return cc.btc_to_currency(btc, time)


def timeToUnix(datetime):
    """
    returns a unix timecode of the input date. Used for frequency analysis.
    """
    return time.mktime(datetime.timetuple())


def checkAddress(blockchain,wallet):
    """
    checks wether a certain address exists in blocksci.
    """
    try:
        if blockchain.address_from_string(wallet):
            return True
    except Exception as e:
        print(e)
        return False


def extract_features_USD(df):
    """
    input: csv of wallet transcations as DataFrame
    output: list of all of our features
    """
    features =  activity_density(df) +  symmetry_score(df) + value_statistics(df)
    return features


def activity_density(df):
    """
    Returns a dict with some time related statistics of the wallet.
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
    Rrturns a list with symmetry related attributes.
    in score/ out score tries to capture the notion of symmetry -
    it calculates how often does input txs are seperated by output txs, and vice versa.
    maximum in both is 0.5, minimum is ( (len(df)^2) ** -1 ) - for regular wallets.
    Receiving only wallets (wallet_type == -1) get a score of the amount of their transactions.
    This should capture the notion of 'sinkholes'.

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
            # Should deprecate
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
    """
    Rrturns a list of value realted statistics - avg transaction value, max fee, etc.
    """
    tx_types = df.tx_type.value_counts().values
    if len(tx_types) == 1:
        if df.tx_type.iloc[0] == 1:
            # Wallet that only receives
            dollar_obtain_per_tx = df.loc[df.tx_type == 1].valueUSD.sum()/df.tx_type.value_counts().values[0]
            dollar_spent_per_tx = 0
            obtain_spent_ratio = None
            wallet_type = -1

        elif df.tx_type.iloc[0] == -1:
            # Wallet that only gives
            # Probably non existant. should deprcate
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
        dollar_obtain_per_tx, # dollar_obtain_per_tx
        dollar_spent_per_tx, # dollar_spent_per_tx
        obtain_spent_ratio, # obtain_spent_ratio
        df.valueUSD.std(), # tx_value_std
        df.feeUSD.max(),  # max_fee
        df.feeUSD.mean(), # avg_fee
        df.shape[0],  # total_num_tx
        df.valueUSD.sum(), # total_dollar
        wallet_type # wallet_type
    ]


def prob_statistics(df):
    """
    Place holder. Once implemented - should join extract_featuresUSD.
    """
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
    pass


def peers_statistics(df):
    # Feature that counts how many distinct peers a wallet have
    # how many close friends does he have (=peers with more than 2 txs)
    pass


def plot_correlations_matrix(df,wanted_method : str,save=True):
    """
    Plots the correlation matrix of all of our features.
    """
    corr_matrix = df.corr(method=wanted_method)
    im = plt.imshow(corr_matrix, cmap=plt.get_cmap('bwr'))
    plt.xticks(np.arange(df.shape[1]-1), df.columns[:-1], rotation=90)
    plt.yticks(np.arange(df.shape[1]-1), df.columns[:-1])
    plt.tight_layout()
    plt.colorbar()
    if save:
        plt.savefig(PATHS.PLOTS_PATH+'Feature_Correlation_matrix.png')
    plt.show()
    plt.close()


def plot_basic_feature_statistics(df,wanted_plot: str,save=False):
    """
    Plots a simpler version of the above function.
    Considering deprecation.
    """
    nrows = df.shape[1]//3
    ncols = 3
    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=(20,20))
    axes = axes.flat
    for i in range(len(axes)):
        print(len(axes))
        df.iloc[:,i].plot(kind=wanted_plot,
                          title=df.columns[i] ,
                          ax=axes[i],
                          bins=100)
    plt.tight_layout()

    if save:
        plt.savefig(PATHS.PLOTS_PATH+'basic_statistics.png')
    plt.show()
    plt.close()


def extract_sum_txs(block):
    """
    returns the sum in USD of all transactions & their fees in a given block.
    Used to calculate distributions for probabilistic features.
    """
    return (BTCtoUSD(sum([tx.input_value for tx in block.txes.to_list()])* PATHS.SATOSHI,block.time),
            BTCtoUSD(sum([tx.fee for tx in block.txes.to_list()]) * PATHS.SATOSHI, block.time),
            block.txes.size)


def extract_range(start,end):
    """
    :param start: First block
    :param end: Last Block
    :return: Returns an array: [avg_tx_usd, avg_fee_usd]
    """
    ti = time.time()
    print(f'Reading block no. {start}..',end='\r')
    temp_array =  sum(np.array(chain.map_blocks(extract_sum_txs,start=start,end=end,cpu_count=4)))
    print(f'Done in {time.time()-ti} seconds.')
    return np.array([temp_array[0]/temp_array[2], temp_array[1]/temp_array[2]])


def make_average_tx_and_fee_dict():
    """
    Returns a dictionary of block numbers as keys (steps of 1000),
            and their distribution of value per tx as values.
            The distribution heuristics needs to be updated before
            the result can be used - average is not a sufficient statistic.
    """
    t = time.time()
    print('Strated extraction..')
    prob_dict = {(1000*k):extract_range((1000*k),((1000*k)+999)) for k in range(0,665)}
    print(f'Done, in {time.time()-t} seconds. Saving..')
    with open('/root/address_book/block_probability_distribution/prob_dict.json','w') as f:
        json.dump(prob_dict,f)
        f.close()
    print('Done. Have a nice day!')




def make_scat_mat_plot():
    fb = get_feature_book(for_ml=True,fraud=True)
    legit_idx = pd.DataFrame(data=[fb.loc[idx] for idx in fb.index if fb.loc[idx]['is_fraud']==False],columns=fb.columns)
    fraud_idx = pd.DataFrame(data=[fb.loc[idx] for idx in fb.index if fb.loc[idx]['is_fraud'] == True],
                             columns=fb.columns)

    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    axes = axes.flat
    axes[0].hist(legit_idx.max_fee.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='blue', alpha=0.5,
                 log=True, label='Legit')
    axes[0].hist(fraud_idx.max_fee.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='red', alpha=0.5,
                 log=True, label='Fraud')
    axes[0].set_title('Max fee')
    axes[0].set_xlabel('Log max_fee')
    axes[0].set_ylabel('Log count')
    axes[0].legend()

    axes[1].hist(legit_idx.total_num_tx, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[1].hist(fraud_idx.total_num_tx, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[1].set_title('Total Num Tx')
    axes[1].set_xlabel('total_num_tx')
    axes[1].set_ylabel('Log count')
    axes[1].legend()

    axes[2].hist(legit_idx.total_dollar, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[2].hist(fraud_idx.total_dollar, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[2].set_title('Total Dollar')
    axes[2].set_xlabel('total_dollar')
    axes[2].set_ylabel('Log count')
    axes[2].legend()

    axes[3].hist(legit_idx.lifetime, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[3].hist(fraud_idx.lifetime, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[3].set_title('Lifetime')
    axes[3].set_xlabel('Days')
    axes[3].set_ylabel('Log count')
    axes[3].legend()

    axes[4].hist(legit_idx.tx_freq_mean, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[4].hist(fraud_idx.tx_freq_mean, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[4].set_title('Tx frequency mean')
    axes[4].set_xlabel('Hours')
    axes[4].set_ylabel('Log count')
    axes[4].legend()

    axes[5].hist(legit_idx.tx_freq_std, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[5].hist(fraud_idx.tx_freq_std, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[5].set_title('Tx frequency standard deviation')
    axes[5].set_xlabel('root(Hours)')
    axes[5].set_ylabel('Log count')
    axes[5].legend()

    axes[6].hist(legit_idx.tx_type_odds.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='blue', alpha=0.5,
                 log=True, label='Legit')
    axes[6].hist(fraud_idx.tx_type_odds.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='red', alpha=0.5,
                 log=True, label='Fraud')
    axes[6].set_title('Tx type odds')
    axes[6].set_xlabel('Odds Ratio')
    axes[6].set_ylabel('Log count')
    axes[6].legend()

    axes[7].hist(legit_idx.consecutive_in_tx_score, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[7].hist(fraud_idx.consecutive_in_tx_score, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[7].set_title('Consecutive incoming txs score')
    axes[7].set_xlabel('Score')
    axes[7].set_ylabel('count')
    axes[7].legend()

    axes[8].hist(legit_idx.consecutive_out_tx_score.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='blue',
                 alpha=0.5, log=True, label='Legit')
    axes[8].hist(fraud_idx.consecutive_out_tx_score.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='red',
                 alpha=0.5, log=True, label='Fraud')
    axes[8].set_title('Consecutive outgoing txs score')
    axes[8].set_xlabel('Score')
    axes[8].set_ylabel('count')
    axes[8].legend()

    axes[9].hist(legit_idx.dollar_obtain_per_tx, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[9].hist(fraud_idx.dollar_obtain_per_tx, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[9].set_title('Avg value of incoming tx')
    axes[9].set_xlabel('USD')
    axes[9].set_ylabel('count')
    axes[9].legend()

    axes[10].hist(legit_idx.dollar_spent_per_tx, bins=50, color='blue', alpha=0.5, log=True, label='Legit')
    axes[10].hist(fraud_idx.dollar_spent_per_tx, bins=50, color='red', alpha=0.5, log=True, label='Fraud')
    axes[10].set_title('Avg value of outgoing tx')
    axes[10].set_xlabel('USD')
    axes[10].set_ylabel('count')
    axes[10].legend()

    axes[11].hist(legit_idx.tx_value_std.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='blue', alpha=0.5,
                  log=True, label='Legit')
    axes[11].hist(fraud_idx.tx_value_std.map(lambda x: np.log(x) if x != 0 else -30), bins=50, color='red', alpha=0.5,
                  log=True, label='Fraud')
    axes[11].set_title('Tx Value Standard deviation')
    axes[11].set_xlabel('root(USD)')
    axes[11].set_ylabel('count')
    axes[11].legend()

    fig.tight_layout()
    fig.savefig('/mnt/plots/scat_mat_plot.png')
    fig.show()
