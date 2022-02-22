import os
import csv
import time
import json
import random
import Analysis
import blocksci
import datetime
import numpy as np
import pandas as pd
from PATHS import *
from multiprocessing import Pool
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor

AGG_DICT = {"valueBTC": "sum",
            "valueUSD": "sum",
            "feeBTC": "sum",
            "feeUSD": "sum",
            "time": "first"}
POOL = Pool(processes=4)


# to do:

# 1. Make hist plot of all elkys features.
# 2. Find a way to describe the distribution of tx fees and values - samples - 1000 blocks.
# 3. Talk to Max about using his email!


class AddressBook:
    def __init__(self):
        self.address_book: dict = self.load_book()

        self.cc = blocksci.currency.CurrencyConverter(currency='USD',
                                                      start=datetime.date(2009, 1, 3),
                                                      end=datetime.date(2021, 1, 31))

        self.services = set([key for key, val in self.address_book.items() if 'services/others' in val])
        self.historic = set([key for key, val in self.address_book.items() if 'old/historic' in val])
        self.exchanges = set([key for key, val in self.address_book.items() if 'exchanges' in val])
        self.gambling = set([key for key, val in self.address_book.items() if 'gambling' in val])
        self.pools = set([key for key, val in self.address_book.items() if 'pools' in val])
        self.fraud = set([key for key, val in self.address_book.items() if 'fraud' in val])

        self.update_addresses = None
        self.large_addresses = set()

        self.found_wallets = set()
        self.found_txes = 0
        self.merged_wallets = 0
        self.merged_lines = 0

        self.extracted_features = 0

        self.vector_dirs = self.get_vector_dirs()
        # self.executor = ThreadPoolExecutor(3)

    @staticmethod
    def load_book():
        with open(ADDRESSBOOK_PATH, 'r') as f:
            book = json.load(f)
            return book

    def make_feature_book(self):
        with open(FEATURE_BOOK_PATH,'w') as f:
            csv.writer(f).writerow(
                ["address","lifetime", "first_tx", "tx_freq_mean", "tx_freq_std", 'tx_type_odds', 'consecutive_in_tx_score',
                 'consecutive_out_tx_score', 'dollar_obtain_per_tx', 'dollar_spent_per_tx', 'obtain_spent_ratio',
                 'tx_value_std', 'max_fee', 'total_num_tx', 'total_dollar', 'wallet_type','tags']
            )

    def write_exrtaction_log(self, e, address):
        t = time.time()
        with open(f'/root/address_book/logs/extraction_logs.csv', 'a') as log:
            csv.writer(log).writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                                      address,
                                      e])
            log.close()

    def write_log(self, block: blocksci.Block):
        t = time.time()
        tot_wallets = len(self.found_wallets)
        with open(f'{ADDRESS_VECTORS_UPDATE}logs.txt', 'a') as log:
            log.write(
                f'\n {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))} '
                f'<- Reached block no.{block.height}, Duration: {time.time() - t}. '
                f'Wallets found: {tot_wallets}. '
                f'Txes found: {self.found_txes} ->')
            log.close()

    def merge_vectors(self):
        t = time.time()
        print(f'Starting merge..')
        # POOL.map(self.merge_single_address, self.update_addresses)
        for address in self.update_addresses:
            self.merge_single_address(address)
        for address in self.large_addresses:
            self.merge_single_address(address, large=True)
        print(
            f'Done merging on {self.merged_wallets} addresses with {self.merged_lines} rows total, in {time.time() - t} seconds.')

    def merge_single_address(self, address: str, large=False):
        if not large:
            locations = self.check_locations(address, self.vector_dirs)
            print(
                f'Merging {address}. So far found {self.merged_lines} unique rows, {self.merged_wallets} wallets, extracted {self.extracted_features} features.',
                end='\r')
            if len(locations) == 0:
                return
            elif len(locations) == 1:
                temp_vector = pd.read_csv(f'{locations[0]}/{address}.csv')
            elif len(locations) > 1:
                temp_vector = [pd.read_csv(f'{loc}/{address}.csv') for loc in locations]
                temp_vector = pd.concat(temp_vector)
            if len(temp_vector) > 5000:
                self.large_addresses.add(address)
                temp_vector.to_csv(LARGE_ADDRESS_VECTORS_PATH + address + '.csv')
                return
        else:
            temp_vector = pd.read_csv(LARGE_ADDRESS_VECTORS_PATH + address + '.csv')
        temp_vector = temp_vector.astype({'valueUSD':np.float64,'feeUSD':np.float64})
        temp_vector.drop_duplicates(inplace=True)
        temp_vector.sort_values(by=['tx_index'], inplace=True)
        temp_vector.groupby(["tx_index", "tx_type", "hash"], as_index=False).agg(AGG_DICT, inplace=True)
        temp_vector = self.updateWalletVector(temp_vector)
        temp_vector.to_csv(ADDRESS_VECTORS_PATH + address + '.csv')
        self.extract_features(address, temp_vector)
        self.merged_lines += len(temp_vector)
        self.merged_wallets += 1

    def extract_features(self, address: str, wallet_df: pd.DataFrame):
        try:
            features = []
            features.append(address)
            features = Analysis.extract_features_USD(wallet_df)
            features.append(self.address_book[address])
            with open(FEATURE_BOOK_PATH, 'a') as f:
                csv.writer(f).writerow(features)
                f.close()
            self.extracted_features += 1
        except Exception as e:
            self.write_exrtaction_log(e, address)

    def get_vector_dirs(self):
        dir_names = sorted([os.path.join('/root/', x) for x in os.listdir('/root') if 'address_vectors_test' in x])
        dir_sets = [set([x for x in os.listdir(dirname) if '.csv' in x]) for dirname in dir_names]
        return list(zip(dir_names, dir_sets))

    def check_locations(self, name, dirs_contents):
        name = f'{name}.csv'
        return [x[0] for x in dirs_contents if name in x[1]]

    def update_range_multiproc(self, addresses, start=None, stop=None):
        chain = blocksci.Blockchain('/root/config.json')
        self.found_wallets = set()
        self.found_txes = 0
        print(f'Running on {stop - start} blocks. First block is {start}.')
        t = time.time()
        chain.map_blocks(self.update_block, start=start, end=stop, cpu_count=4)
        print(
            f'Done in {time.time() - t} seconds. Found a total of {self.found_wallets} wallets and {self.found_txes} txes.')

    def update_range(self, addresses, start=None, stop=None):
        """
        Creates or updates .csv file containing tx information of the selected addresses.
        :param addresses: Array of addresses.
        :param start: First block
        :param stop: Last block
        :return:
        """
        chain = blocksci.Blockchain('/root/config.json')

        self.update_addresses = set(addresses)
        self.found_wallets = set()
        self.found_txes = 0

        if start and stop:
            print(f'Running on {stop - start} blocks. First block is {start}.')
            t = time.time()
            [self.update_block(block) for block in chain.blocks[start:stop]]

        elif start and (not stop):
            print(f'Running on all blocks from {start}.')
            t = time.time()
            [self.update_block(block) for block in chain.blocks[start:]]

        elif (not start) and stop:
            print(f'Running on all blocks up to {stop}.')
            t = time.time()
            [self.update_block(block) for block in chain.blocks[:stop]]

        else:
            print(f'Running on all blocks. Good luck my friend.')
            t = time.time()
            [self.update_block(block) for block in chain.blocks]

        print(
            f'Done in {time.time() - t} seconds. Found a total of {self.found_wallets} wallets and {self.found_txes} txes.',
            end='\r')

    def update_block(self, block: blocksci.Block):
        """
        Iterates over the txes in a single block and writes them to the
        desired csv files.
        :param block:
        """
        print(f'Reading block: {block.hash.__str__()}. Txes found so far: {self.found_txes}.', end='\r')
        t = time.time()
        if block.height % 5000 == 0:
            print(f'Reached block no.{block.height}')
            self.write_log(block)
        [self.write_tx(add[0], add[1], add[2], idx, tx) for idx, tx in enumerate(block.txes.to_list()) for add in
         self.tx_to_address_list(tx)]
        print(f'Done in {time.time() - t} seconds.', end='\r')

    def write_tx(self, address: str, tx_type: int, address_idx_in_tx: int, tx_idx_in_block: int, tx: blocksci.Tx):
        """
        :param address: Bitcoin Wallet address
        :param tx_type: Input / Output , position of wallet in the tx (±1)
        :param address_idx_in_tx: index of address in the current transcation,
               used to extract the amount being transacted to/from this address.
        :param tx_idx_in_block: index of the tx in the block.
        :param tx: the Tx object.
        :return: no return - updates the
        """

        self.found_txes += 1
        if not address in self.found_wallets:
            self.make_wallet_vector(address)
            self.found_wallets.add(address)
            print(f'Added wallet vector for {address}', end='\r', )
        try:
            tx_value = self.get_value(tx, tx_type, address_idx_in_tx)
            with open(os.path.join(ADDRESS_VECTORS_UPDATE, address + '.csv'), 'a') as f:
                csv.writer(f).writerow([tx_type,
                                        tx_value,
                                        0,
                                        (tx_value / tx.input_value) * tx.fee if tx_type == -1 else 0,
                                        0,
                                        tx.block_time,
                                        tx.hash.__str__(),
                                        tx.index])
            print(f'Added {address} in tx no.{tx_idx_in_block}', end='\r')
        except Exception as e:
            print(e)

    def make_wallet_vector(self, address: str):
        """
        Creates an empty .csv file for the desired address.
        """
        with open(os.path.join(ADDRESS_VECTORS_UPDATE, address + '.csv'), 'w') as f:
            csv.writer(f).writerow(['tx_type', 'valueBTC', 'valueUSD', 'feeBTC', 'feeUSD', 'time', 'hash', 'tx_index'])
            f.close()

    def get_value(self, tx: blocksci.Tx, tx_type: int, index: int):
        """
        :param tx: Tx object
        :param type: input / output, ±1
        :param index: index of wallet in this tx.
        :return: The amount transacted in/out of this wallet.
        """
        try:
            if tx_type == -1:
                return tx.ins.value[index] * SATOSHI
            elif tx_type == 1:
                return tx.outs.value[index] * SATOSHI
        except Exception as e:
            print(e)

    def tx_to_address_list(self, tx: blocksci.Tx):
        """
        :param tx: Blocksci tx
        :return: list of tuples of 3: address,
                                      type (±1)
                                      index in tx (used for value extraction)
        """
        if hasattr(tx.ins.address, 'to_list'):
            ins_list = [(address.address_string, -1, in_idx) for in_idx, address in enumerate(tx.ins.address.to_list())
                        if (hasattr(address, 'address_string') and address.address_string in self.update_addresses)]
            outs_list = [(address.address_string, 1, out_idx) for out_idx, address in
                         enumerate(tx.outs.address.to_list())
                         if (hasattr(address, 'address_string') and address.address_string in self.update_addresses)]
        return ins_list + outs_list

    # def multi_tx_to_address_list(self, tx: blocksci.Tx):
    #     """
    #     :param tx: Blocksci tx
    #     :return: list of tuples of 3: address,
    #                                   type (±1)
    #                                   index in tx (used for value extraction)
    #     """
    #     if hasattr(tx.ins.address, 'to_list'):
    #         ins_list = [self.pool.map(lambda idx, address: (address.address_string, -1, idx)
    #         if (hasattr(address, 'address_string') and address.address_string in self.update_addresses) else None
    #                                   , enumerate(tx.ins.address.to_list()))
    #                     ]
    #         outs_list = [self.pool.map(lambda idx, address: (address.address_string, 1, idx)
    #         if (hasattr(address, 'address_string') and address.address_string in self.update_addresses) else None
    #                                    , enumerate(tx.outs.address.to_list()))
    #                      ]
    #     return ins_list + outs_list

    def updateWalletVector(self, wallet_vector: pd.DataFrame):
        """
        :param wallet_vector: A pandas dataframe holding the timeseries of a wallet.
        :return: no return. updates the ValueUSD column.
        """
        if len(wallet_vector) == 0:
            pass
        else:
            for idx in range(len(wallet_vector)):
                if wallet_vector.iloc[idx]['valueUSD'] == 0.0:
                    wallet_vector.iat[idx, 2] = self.cc.btc_to_currency(wallet_vector.iloc[idx]['valueBTC'],
                                                                        wallet_vector.iloc[idx]['time'])
                    wallet_vector.iat[idx, 4] = self.cc.btc_to_currency(wallet_vector.iloc[idx]['feeBTC'],
                                                                        wallet_vector.iloc[idx]['time'])
                    wallet_vector.iat[idx, 5] = self.timeToUnix(wallet_vector.iloc[idx]['time'])
        return wallet_vector

    @staticmethod
    def timeToUnix(date_time):
        return time.mktime(datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S").timetuple())

    def plot_wallet_vector(self, address: str, wallet_vector: pd.DataFrame, size: float, save=False, wallet_tags=None,
                           symmetry=True):
        """
        Plots the tx value of every tx of the desired wallet, over time.
        Inputs are colored blue, outputs in red.
        """
        plt.close()
        if symmetry:
            wallet_vector.valueUSD = np.multiply(wallet_vector.valueUSD, wallet_vector.tx_type)
        scatter = plt.scatter(wallet_vector.time,
                              wallet_vector.valueUSD,
                              c=wallet_vector.tx_type,
                              cmap='coolwarm',
                              s=size)
        plt.suptitle('Transcations over time')
        plt.ylabel('Tx Value USD')
        plt.xlabel('Time')
        if wallet_tags:
            tags = ''
            for tag in wallet_tags:
                tags = tags + ', ' + tag
            plt.title(f'Wallet tags: {tags}')
            plt.gca().add_artist(plt.legend([address], loc=4))
        plt.legend(handles=scatter.legend_elements()[0], labels=['Input', 'Output'])
        if save:
            filename = f'{PLOTS_PATH}AV_{address}.png'
            plt.savefig(filename)
        else:
            plt.show()


def test_n_times_multi(n, start, stop):
    test_results = []
    for test in range(n):
        test_results.append(test_multi_update(start, stop))
    return test_results


def test_n_times(n, start, stop):
    test_results = []
    for test in range(n):
        test_results.append(test_update(start, stop))
    return test_results


def test_update(start, stop, checkpoint=None):
    ab = AddressBook()
    t = time.time()
    if checkpoint:
        ab.found_wallets = set([str(f.split('.csv')[0]) for f in os.listdir(ADDRESS_VECTORS_UPDATE)])
        print(f'Starting with {len(ab.found_wallets)}.')
        ab.update_range(ab.address_book.keys(), start=checkpoint, stop=stop)
    else:
        ab.update_range(ab.address_book.keys(), start=start, stop=stop)
    print(f'Total time for 100 blocks:{time.time() - t}')
    return time.time() - t


def test_multi_update(start, stop):
    ab = AddressBook()
    t = time.time()
    ab.update_range_multiproc(ab.address_book.keys(), start=start, stop=stop)
    print(f'Total time for 100 blocks:{time.time() - t}')
    return time.time() - t


def test_merge(test_set=None):
    ab = AddressBook()
    if not test_set:
        ab.update_addresses = set(random.sample(ab.address_book.keys(), 100))
        test_set = ab.update_addresses
    else:
        ab.update_addresses = test_set
    print('First run:')
    ab.merge_vectors()
    print('Second run:')
    ab.merge_vectors()
    print('Third run:')
    ab.merge_vectors()
    return test_set


def merge_all():
    ab = AddressBook()
    found_addresses = set([address.replace('.csv', '') for address in os.listdir(ADDRESS_VECTORS_PATH)])
    ab.update_addresses = set(ab.address_book.keys()).difference(set(found_addresses))
    ab.merge_vectors()


if __name__ == '__main__':
    merge_all()