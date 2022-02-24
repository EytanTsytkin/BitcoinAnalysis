import os
import bs4
import time
import json
import requests
import pandas as pd
import csv

from PATHS import *

BASE = 'https://www.walletexplorer.com/'
# TAGS_PATH = 'F:\DataS-HUJI\FinalProject\CSVFromWebsite\\tags\\'
TAGS_PATH = 'F:\DataS-HUJI\FinalProject\CSVFromWebsite\\lost\\'
ADDRESSBOOK_PATH = '..\JsonFile\AddressBook.json'
DOWNLOAD ='addresses?format=csv'

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
    # filename = f'{TAGS_PATH}tags{name.lower()}.csv' match to the first TAGS_PATH(tags)
    filename = f'{TAGS_PATH}{name.lower()}.csv'
    try:
        data = pd.read_csv(filename,header=1)
        print(f'{name} success, adding to address book.\r')
    except:
        print(f'{name} failed, proceeding..\r')
        return []
    return [address for address in data['address']]

def getNameCsv(name,filename):
    time.sleep(1)
    file_old = f'{TAGS_PATH}tags{name.lower()}.csv'
    if os.path.isfile(file_old):
        url = f'{BASE}wallet/{name}/{DOWNLOAD}'
        print(f'Getting {name}..\r')
        res = requests.get(url)
        with open(filename, 'wb') as i:
            i.write(res.content)

def getData(Typedict):
    if not Typedict:
        Typedict = makeTypeSitesDict()
    for key in Typedict.keys():
        for name in Typedict[key]:
            filename = f'{TAGS_PATH}{name.lower()}.csv'
            if not os.path.isfile(filename):
                getNameCsv(name, filename)

def makeAddressBook():
    Typedict = makeTypeSitesDict()
    file = open(ADDRESSBOOK_PATH, 'r')
    AddressBook = eval(file.read())
    # getData(Typedict) get the 118 lost files
    for key in Typedict.keys():
        for name in Typedict[key]:
            for addr in nameToAddressList(name):
                AddressBook[addr] = (key,name)
    return AddressBook


def createJsonFile():
    addressBook = makeAddressBook()
    with open(ADDRESSBOOK_PATH,'w') as i:
        print(f'Success. Saving...')
        json.dump(addressBook,i)
        print(f'Done.')

createJsonFile()