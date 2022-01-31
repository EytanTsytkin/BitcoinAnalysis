import json

def addressBook():
    with open('/mnt/address_tags/AddressBook.json','r') as i:
        address_book = json.loads(i.read())
    return address_book
