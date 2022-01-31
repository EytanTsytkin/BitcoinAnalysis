import os
import json
import requests
import pandas as pd
import time
import bs4

#WalletExplorer.com
BASE = 'https://www.walletexplorer.com/'
DOWNLOAD ='addresses?format=csv'
ADDRESSBOOK_PATH = '/mnt/address_tags/AddressBook.json'


def makeTypeSitesDict():
    # returns a dictionary with types (exchanges,pools..) as keys,
    # with list of sites of the corresponding type as values.
    res = requests.get(BASE)
    soup = bs4.BeautifulSoup(res.text,'html.parser')
    parents = soup.findAll('td')
    return {p.findChild('h3').text.replace(':',"").lower() : [c.text.split(' ')[0]
                                                                  for c in p.findChildren('li')]
                                                                                for p in parents }


def nameToAddressList(name):
    filename = f'/mnt/address_tags/{name.lower()}.csv'
    try:
        data = pd.read_csv(filename,header=1)
        print(f'{name} success, adding to address book.\r')
    except:
        print(f'{name} failed, proceeding..\r')
        return []
    return [address for address in data['address']]


def getNameCsv(name,filename):
    time.sleep(1)
    url = f'{BASE}wallet/{name}/{DOWNLOAD}'
    print(f'Getting {name}..\r')
    res = requests.get(url)
    with open(filename, 'wb') as i:
        i.write(res.content)


def getData(Typedict):
    corrupt_files = testData()
    if not Typedict:
        Typedict = makeTypeSitesDict()
    for key in Typedict.keys():
        for name in Typedict[key]:
            filename = f'/mnt/address_tags/{name.lower()}.csv'
            if not os.path.isfile(filename) or name.lower() in corrupt_files:
                getNameCsv(name, filename)


def testData():
    count = 0
    corrupt_files = set()
    for file in os.listdir('/mnt/address_tags/'):
        try:
            data = pd.read_csv(f'/mnt/address_tags/{file}', header=1)
        except:
            count +=1
            corrupt_files.add(file.split('.csv')[0])
    print(f'Found {count} corrupt files.')
    return corrupt_files


def makeAddressBook(Typedict):
    if not Typedict:
        Typedict = makeTypeSitesDict()
    getData(Typedict)
    AddressBook = dict()
    for key in Typedict.keys():
        for name in Typedict[key]:
            for addr in nameToAddressList(name):
                AddressBook[addr] = (key,name)
    return AddressBook




