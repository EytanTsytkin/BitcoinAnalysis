import csv
import json

#https://blockchair.com/
# addresses: 30000
Typedict = {'exchanges': ['blockchair']}
ADDRESSBOOK_FILE = '..\JsonFile\AddressBook.json'

def makeAddressBookFromCSV():
    file = open(ADDRESSBOOK_FILE, 'r')
    AddressBook = eval(file.read())
    address_list = []
    with open('..\OtherFiles\blockchair_addresses.tsv', 'rt') as csvfile:
        reader = csv.reader(csvfile,delimiter=',',quotechar='"',doublequote=False)
        address_list = [row[0] for row in reader]
    for key in Typedict.keys():
        for name in Typedict[key]:
            for addr in address_list:
                AddressBook[addr] = (key, name)
    with open(ADDRESSBOOK_FILE,'w') as i:
            json.dump(AddressBook,i)


getAddressesFromCSV()
