from PATHS import *
import json
import blocksci
import numpy as np
import pandas as pd

class AddressBook:
    def __init__(self):
        self.address_book: dict = self.load_book()
        self.update_addresses = set([self.address_book.keys()])


    @staticmethod
    def load_book():
        with open('/root/address_book/AddressBook.json', 'r') as f:
            book = json.load(f)
        dictionary = {}
        for wallet, wallet_list in book.items():
            dictionary[wallet] = {
                'type': wallet_list[0],
                'specific_type': wallet_list[1],
                'txes' : [],
                'wallet_vector': pd.DataFrame(columns=['type', 'value', 'time'])
            }
        return dictionary

    def update_book(self, block: blocksci.Block, save = False):
        for tx in block.txes.to_list():
            ins,outs = self._tx_to_address_list(tx)
            for type in [ins,outs]:
                for idx,address in enumerate(type):
                    if address[0] in self.update_addresses:
                        self.address_book[address][wallet_vector].append({'type':address[1],
                                                                          'valiue':self.get_value(tx,'in',address),
                                                                          'time':tx.block_time})


    def get_value(tx: blocksci.Tx ,type:str, index: int):
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


def main():
    ab = AddressBook()
    print(ab)

main()