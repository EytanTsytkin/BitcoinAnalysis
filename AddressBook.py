import csv
import os
import time
import json
import random
import blocksci
import datetime
import numpy as np
import pandas as pd
from PATHS import *
from  multiprocessing import Pool
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor


#LALALA

ADDRESS_VECTORS_PATH = '/root/address_vectors_test380k/'

class AddressBook:
    def __init__(self):
        self.address_book: dict = self.load_book()
        self.update_addresses = None
        self.cc = blocksci.currency.CurrencyConverter(currency='USD',
                                                      start=datetime.date(2009, 1, 3),
                                                      end=datetime.date(2021, 1, 31))
        self.found_wallets = set()
        self.found_txes = 0
        #self.pool = Pool(processes=16)
        #self.executor = ThreadPoolExecutor(3)


    @staticmethod
    def load_book():
        with open(ADDRESSBOOK_PATH, 'r') as f:
            book = json.load(f)
            return book

    def backup(self,block: blocksci.Block):
        t = time.time()
        tot_wallets = len(self.found_wallets)
        # for key, val in self.address_book.items():
        #     if 'wallet_vector' in val.keys() and type(val['wallet_vector']) == pd.DataFrame:
        #         val['wallet_vector'] = self.updateWalletVector(val['wallet_vector'])
        #         with open('/mnt/address_vectors4/' + str(key) + '.csv', 'w') as f:
        #             val['wallet_vector'].to_csv(f)
        with open('/root/address_vectors_test380k/logs.txt', 'a') as log:
            log.write(
                f'\n {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))} '
                f'<- Reached block no.{block.height}, Duration: {time.time()-t}. '
                f'Wallets found: {tot_wallets}. '
                f'Txes found: {self.found_txes} ->')


    def update_range_multiproc(self,addresses,start=None,stop=None):
        self.update_addresses = set(addresses)
        chain = blocksci.Blockchain('/root/config.json')
        self.found_wallets = set()
        self.found_txes = 0
        print(f'Running on {stop - start} blocks. First block is {start}.')
        t = time.time()
        chain.map_blocks(self.update_block,start=start,end=stop,cpu_count=4)
        print(f'Done in {time.time() - t} seconds. Found a total of {self.found_wallets} wallets and {self.found_txes} txes.')


    def update_range(self,addresses,start=None,stop=None):
        self.update_addresses = set(addresses)
        chain = blocksci.Blockchain('/root/config.json')
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
        print(f'Done in {time.time() - t} seconds. Found a total of {self.found_wallets} wallets and {self.found_txes} txes.',end='\r')


    def update_block(self, block: blocksci.Block):
        print(f'Reading block: {block.hash.__str__()}. Txes found so far: {self.found_txes}.', end='\r')
        t = time.time()
        if block.height % 5000 == 0:
            print(f'Reached block no.{block.height}')
            self.backup(block)
        [self.write_tx(add[0], add[1], add[2], idx, tx) for idx, tx in enumerate(block.txes.to_list()) for add in
             self.tx_to_address_list(tx)]
        print(f'Done in {time.time()-t} seconds.',end='\r')


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
            with open(os.path.join(ADDRESS_VECTORS_PATH, address + '.csv'), 'a') as f:
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
        with open(os.path.join(ADDRESS_VECTORS_PATH,address+'.csv'),'w') as f:
            csv.writer(f).writerow(['tx_type', 'valueBTC', 'valueUSD', 'feeBTC', 'feeUSD', 'time', 'hash','tx_index'])
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
        elif len(wallet_vector) == 1:
            if wallet_vector.iloc[0]['valueUSD'] == 0:
                wallet_vector.iat[0, 2] = self.cc.btc_to_currency(wallet_vector.iloc[0]['valueBTC'],
                                                                  wallet_vector.iloc[0]['time'])
                wallet_vector.iat[0, 4] = self.cc.btc_to_currency(wallet_vector.iloc[0]['feeBTC'],
                                                                  wallet_vector.iloc[0]['time'])
                wallet_vector.iat[0, 5] = self.timeToUnix(wallet_vector.iloc[0]['time'])

        elif len(wallet_vector) > 1:
            for idx in range(1, len(wallet_vector)):
                if wallet_vector.iloc[idx]['valueUSD'] == 0:
                    wallet_vector.iat[idx, 1] = (wallet_vector.iloc[idx]['valueBTC'] - wallet_vector.iloc[idx - 1]['valueBTC'])
                    wallet_vector.iat[idx, 2] = self.cc.btc_to_currency(wallet_vector.iloc[idx]['valueBTC'],
                                                                        wallet_vector.iloc[idx]['time'])
                    wallet_vector.iat[idx, 4] = self.cc.btc_to_currency(wallet_vector.iloc[idx]['feeBTC'],
                                                                      wallet_vector.iloc[idx]['time'])
                    wallet_vector.iat[idx, 5] = self.timeToUnix(wallet_vector.iloc[idx]['time'])
        return wallet_vector

    @staticmethod
    def timeToUnix(datetime):
        return time.mktime(datetime.timetuple())


    def plotValueTimeSeries(self, address, wallet_vector, size, save=False,wallet_type=None):
        plt.close()
        scatter = plt.scatter(wallet_vector["time"],
                              wallet_vector["valueUSD"],
                              c=wallet_vector["type"],
                              cmap='coolwarm',
                              s=size)
        if wallet_type:
            plt.title(f'Tx over time in {wallet_type}')
            plt.gca().add_artist(plt.legend([address], loc=4))
        else:
            plt.title(f'Tx over time in {address}')
        plt.xlabel('Time')
        plt.ylabel('Tx Value USD')
        plt.legend(handles=scatter.legend_elements()[0], labels=['Input', 'Output'])
        if save:
            filename = f'{PLOTS_PATH}AV_{address}.png'
            plt.savefig(filename)
        else:
            plt.show()


def test_n_times_multi(n,start,stop):
    test_results = []
    for test in range(n):
        test_results.append(test_multi_update(start, stop))
    return test_results


def test_n_times(n,start,stop):
    test_results = []
    for test in range(n):
        test_results.append(test_update(start,stop))
    return test_results

def test_update(start,stop,checkpoint=None):
    ab = AddressBook()
    t =time.time()
    if checkpoint:
        ab.found_wallets=set([str(f.split('.csv')[0]) for f in os.listdir(ADDRESS_VECTORS_PATH)])
        print(f'Starting with {len(ab.found_wallets)}.')
        ab.update_range(ab.address_book.keys(), start=checkpoint, stop=stop)
    else:
        ab.update_range(ab.address_book.keys(),start=start,stop=stop)
    print(f'Total time for 100 blocks:{time.time()-t}')
    return time.time()-t


def test_multi_update(start,stop):
    ab = AddressBook()
    t =time.time()
    ab.update_range_multiproc(ab.address_book.keys(),start=start,stop=stop)
    print(f'Total time for 100 blocks:{time.time()-t}')
    return time.time()-t

test_update(380000,None)
# # Results for blocks 190000-190100, single thread
# res1 = [57.16300082206726, 57.30099153518677, 57.65855407714844]
#
# # Results for blocks 190000-190100, single thread, no list comprehension for blocks:
# res2 = [57.13825750350952, 56.91519618034363, 57.32981038093567]
#
# # Results for blocks 190000-190100, 16 threads:
# res3 = [68.33737421035767, 68.18032550811768, 68.81083369255066]
#
# # Results for blocks 190000-190100, 3 threads:
# res4 = [66.5042371749878, 66.51269555091858, 66.48649406433105]
#
# # Results for blocks 190000-190100, 4 cpu, blocksci. map blocks:
# res5 = [28.303866624832153, 27.49988317489624, 27.342219591140747]
#
# # Results for blocks 190000-190100, single thread with higher open files limit-
# #res6 = similar to res 1, slightly better
#
# # Results for blocks 190000-190100, 8 cpu, blocksci. map blocks:
# res7 = [32.515013456344604, 32.351489782333374, 32.329580545425415]
#
# # Results for blocks 190000-190100, 2 cpu, blocksci. map blocks:
# res8 = [34.220452070236206, 33.99496340751648, 33.794970989227295]
#
# # Results for blocks 190000-190100, 5 cpu, blocksci. map blocks:
# res9 = [28.276405334472656, 29.267545700073242, 28.557278871536255]
#
# # Results for blocks 190000-190100, 4 cpu, blocksci. map blocks, writecsv instead of memory:
# res10 = [16.83560299873352]
#
# # Results for blocks 190000-190100, 1 cpu, writecsv instead of memory:
# res11 = [17.29053258895874]
#
# # Results for blocks 190000-190100, 8 cpu, blocksci. map blocks, writecsv instead of memory:
# res12 = [25.467056035995483, 25.344855785369873, 25.35331916809082]

