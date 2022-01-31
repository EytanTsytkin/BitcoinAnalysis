import time
import json
import blocksci
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SATOSHI = 10**-8
PLOTS_PATH = '/mnt/plots/'

chain = blocksci.Blockchain("/root/BlockSci/config.json")
cc = blocksci.currency.CurrencyConverter(currency='USD',
                                         start=datetime.date(2009,1,3),
                                         end=datetime.date(2017,1,25))
addr = blocksci.address_type.pubkey
wallet = chain.address_from_string(chain.addresses(addr)[2426948].address_string)


def timeToUnix(datetime):
    return time.mktime(datetime.timetuple())


def BTCtoUSD(btc,time):

    return cc.btc_to_currency(btc, time)


### MUST BE FIXED - we need to
def address_Vector(wallet):
    # Returns a time series of wallet balance in USD
    # Each timestamp is a tx
    # 1 = wallet receives money (it is an output of the tx)
    # -1 = wallet sends money (it is an input of the tx)
    out_list = [(1,
                 cc.btc_to_currency(wallet.balance(tx.block_height) * SATOSHI,tx.block_time),
                 timeToUnix(tx.block_time))for tx in wallet.output_txes]
    in_list = [(-1,
                cc.btc_to_currency(wallet.balance(tx.block_height) * SATOSHI,tx.block_time),
                timeToUnix(tx.block_time)) for tx in wallet.input_txes]
    res = sorted(out_list + in_list,key=lambda x: x[2])
    res = pd.DataFrame(res,columns=["type","value","time"])
    return res


def VT_VecScore(VTVec):
    pass



def plotValueTimeSeries(address,timeSeries,size,save=False):
    plt.close()
    scatter = plt.scatter(timeSeries["time"],
                          timeSeries["value"],
                          c=timeSeries["type"],
                          cmap='coolwarm',
                          s=size)
    plt.title(f'Value over time in {address}')
    plt.xlabel('Time')
    plt.ylabel('Value USD')
    plt.legend(handles=scatter.legend_elements()[0],labels=['Input','Output'])
    if save:
        filename = f'{PLOTS_PATH}VOT_{address}.png'
        plt.savefig(filename)
    else:
        plt.show()