import os
import time
import json
import random
import blocksci
import datetime
import numpy as np
import pandas as pd
from PATHS import *


class AddressBook:
    def __init__(self):
        self.address_book: dict = self.load_book()
        self.update_addresses = None
        self.cc = blocksci.currency.CurrencyConverter(currency='USD',
                                        start=datetime.date(2009,1,3),
                                         end=datetime.date(2021,1,31))

    @staticmethod
    def load_book():
        with open(ADDRESSBOOK_PATH, 'r') as f:
            book = json.load(f)
            return book
        # dictionary = {}
        # for wallet, wallet_list in book.items():
        #     dictionary[wallet] = {
        #         'type': wallet_list[0],
        #         'specific_type': wallet_list[1],
        #         'txes' : [],
        #         'wallet_vector': pd.DataFrame(columns=['type', 'valueBTC','valueUSD', 'time'])
        #     }
        # return dictionary
    def

    def update_block(self, block: blocksci.Block, save = False):
        t = time.time()
        print(f'Reading block: {block.hash.__str__()}',end='\r')
        if block.height%2500 == 0:
            print(f'Reached block no.{block.height}')
            count = 0
            for key,val in self.address_book.items():
                if 'wallet_vector' in val.keys() and type(val['wallet_vector']) ==  pd.DataFrame:
                    count += 1
                    with open('/mnt/address_vectors/'+str(key)+'.csv','w') as f:
                        val['wallet_vector'].to_csv(f)
            with open('/mnt/address_vectors/logs.txt','a') as log:
                log.write(f'\n {time.time()} --- Reached block no.{block.height}, Duration: {time.time()-t}. Total address with wallet vectors: {count} ---')
        for idx, tx in enumerate(block.txes.to_list()):
            ins_outs_list = self.tx_to_address_list(tx)
            [self.write_tx(add[0],add[1],add[2],idx,tx) for add in ins_outs_list]
            #for address_tuple in ins_outs_list:
                    # self.write_tx(address[0],address[1],indx,idx,tx)
                    # print(f'Found {address[0]} in tx no.{idx}')
                    # self.address_book[address[0]]['wallet_vector'] = self.address_book[address[0]]['wallet_vector'].append({'type':address[1],
                    #                 'valueBTC':self.get_value(tx,address[1],indx),
                    #                 'valueUSD':0,
                    #                 'time':tx.block_time},
                    #                 ignore_index=True)


    def write_tx(self, address: str,tx_type: int, address_idx_in_tx: int, tx_idx_in_block: int ,tx: blocksci.Tx):
        """
        :param address: Bitcoin Wallet address
        :param tx_type: Input / Output , position of wallet in the tx (±1)
        :param address_idx_in_tx: index of address in the current transcation,
               used to extract the amount being transacted to/from this address.
        :param tx_idx_in_block: index of the tx in the block.
        :param tx: the Tx object.
        :return: no return - updates the
        """
        print(f'Found {address} in tx no.{tx_idx_in_block}', end='\r')
        try:
            if 'wallet_vector' in self.address_book[address].keys() and self.address_book[address]['wallet_vector'] is not None:
                tx_value = self.get_value(tx, tx_type, address_idx_in_tx)
                self.address_book[address]['wallet_vector'] = self.address_book[address]['wallet_vector'].append(
                    {'type': tx_type,
                     'valueBTC': tx_value,
                     'valueUSD': 0,
                     'feeBTC:': (tx_value/tx.input_value) * tx.fee if tx_type == -1 else 0,
                     'feeUSD': 0,
                     'time': tx.block_time,
                     'hash':tx.hash.__str__()},
                    ignore_index=True)
            else:
                self.make_wallet_vector(address)
                print(f'Added wallet vector for {address}', end='\r')
                tx_value = self.get_value(tx, tx_type, address_idx_in_tx)
                self.address_book[address]['wallet_vector'] = self.address_book[address]['wallet_vector'].append(
                    {'type': tx_type,
                     'valueBTC': tx_value,
                     'valueUSD': 0,
                     'feeBTC:': (tx_value/tx.input_value) * tx.fee if tx_type == -1 else 0,
                     'feeUSD': 0,
                     'time': tx.block_time,
                     'hash':tx.hash.__str__()},
                    ignore_index=True)
        except Exception as e:
            print(e)


    def make_wallet_vector(self, address: str):
        self.address_book[address]['wallet_vector'] = pd.DataFrame(columns=['type', 'valueBTC','valueUSD','feeBTC','feeUSD','time','hash'])

    def get_value(self,tx: blocksci.Tx ,tx_type:int, index: int):
        """
        :param tx: Tx object
        :param type: input / output, ±1
        :param index: index of wallet in this tx.
        :return: The amount transacted in/out of this wallet.
        """
        try:
            if tx_type ==  -1:
                return tx.ins.value[index] * SATOSHI
            elif tx_type == 1:
                return tx.outs.value[index] * SATOSHI
        except Exception as e:
            print(e)


    def tx_to_address_list(self,tx: blocksci.Tx):
        """
        :param tx: Blocksci tx
        :return: list of tuples of 3: address,
                                      type (±1)
                                      index in tx (used for value extraction)

        """
        if hasattr(tx.ins.address, 'to_list'):
            ins_list = [(address.address_string,-1,in_idx) for in_idx, address in enumerate(tx.ins.address.to_list())
                        if (hasattr(address, 'address_string') and address.address_string in self.update_addresses)]
            outs_list = [(address.address_string,1,out_idx) for out_idx, address in enumerate(tx.outs.address.to_list())
                         if (hasattr(address, 'address_string') and address.address_string in self.update_addresses)]
        return ins_list+outs_list


    def updateWalletVector(self,wallet_vector: pd.DataFrame):
        """
        :param wallet_vector: A pandas dataframe holding the timeseries of a wallet.
        :return: no return. updates the ValueUSD column.
        """
        if len(wallet_vector) == 0:
            pass
        elif len(wallet_vector) == 1:
            if wallet_vector.iloc[0]['valueUSD'] == 0:
                wallet_vector.iat[0,2] = self.cc.btc_to_currency(wallet_vector.iloc[0]['valueBTC'], wallet_vector.iloc[0]['time'])
                wallet_vector.iat[0,3] = self.timeToUnix(wallet_vector.iloc[0]['time'])
        elif len(wallet_vector) > 1:
            for idx in range(1, len(wallet_vector)):
                if wallet_vector.iloc[idx]['valueUSD'] == 0:
                    wallet_vector.iat[idx,1] = wallet_vector.iloc[idx]['valueBTC'] - wallet_vector.iloc[idx-1]['valueBTC']
                    wallet_vector.iat[idx,2] = self.cc.btc_to_currency(wallet_vector.iloc[idx]['valueBTC'], wallet_vector.iloc[idx]['time'])
                    wallet_vector.iat[idx,3] = self.timeToUnix(wallet_vector.iloc[idx]['time'])


    @staticmethod
    def timeToUnix(datetime):
        return time.mktime(datetime.timetuple())



def test_update():
    ab = AddressBook()
    ab.update_addresses = set(ab.address_book.keys())
    chain = blocksci.Blockchain('/root/config.json')
    [ab.update_block(block) for block in chain.blocks]
    return ab

ab = test_update()

