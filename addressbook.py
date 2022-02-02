import json

def addressBook():
    with open('/root/address_book/AddressBook.json','r') as i:
        address_book = json.loads(i.read())
    return address_book
