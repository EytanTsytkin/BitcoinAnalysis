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
        self.cc = cc = blocksci.currency.CurrencyConverter(currency='USD',
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


    def update_block(self, block: blocksci.Block, save = False):
        print(f'Reading block: {block.hash.__str__()}',end='\r')
        for idx, tx in enumerate(block.txes.to_list()):
            ins,outs = self._tx_to_address_list(tx)
            for lst in [ins,outs]:
                for indx,address in enumerate(lst):
                    if address[0] in self.update_addresses:
                        print(f'Found {address[0]} in tx no.{idx}')
                        self.address_book[address[0]]['wallet_vector'] = self.address_book[address[0]]['wallet_vector'].append({'type':address[1],
                                        'valueBTC':self.get_value(tx,address[1],indx),
                                        'valueUSD':0,
                                        'time':tx.block_time},
                                        ignore_index=True)



    def get_value(self,tx: blocksci.Tx ,type:int, index: int):
        if type ==  -1:
            return tx.ins.value[index] * SATOSHI
        elif type == 1:
            return tx.outs.value[index] * SATOSHI


    @staticmethod
    def _tx_to_address_list(tx):
        ins_list = []
        outs_list = []
        if hasattr(tx.ins.address, 'to_list'):
            for address in tx.ins.address.to_list():
                if hasattr(address, 'address_string'):
                    ins_list.append(address.address_string)
        if hasattr(tx.outs.address, 'to_list'):
            for address in tx.outs.address.to_list():
                if hasattr(address, 'address_string'):
                    outs_list.append(address.address_string)

        return zip(ins_list,[-1 for i in ins_list])  , zip(outs_list,[1 for o in outs_list])


    def updateWalletVector(self,wallet_vector):
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



def test_update(n):
    ab = AddressBook()
    ab.update_addresses = set(random.sample(ab.address_book.keys(), n))
    chain = blocksci.Blockchain('/root/config.json')
    b = [chain.address_from_string(ad).first_tx.block.hash.__str__() for ad in ab.update_addresses]
    blocks = list(filter((lambda block: block.hash.__str__() in b), chain.blocks.to_list()))
    for block in blocks:
        ab.update_block(block)
    for ad in ab.update_addresses:
        print(ab.address_book[ad]['wallet_vector'])
    for ad in ab.update_addresses:
        ab.updateWalletVector(ab.address_book[ad]['wallet_vector'])
        print(ab.address_book[ad]['wallet_vector'])
    return ab

def test_single_wallet()