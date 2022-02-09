import json

def loadBook():
    with open('/root/address_book/AddressBook.json','r') as i:
        book = json.loads(i.read())
    return book


class Addressbook():
    def __init__(self,book,group=None,subgroup=None):
        self.allTree = {val[0]:val[1] for val in book.values()}
        try:
            if group and subgroup:
                self.book = set([key for key,val in book.items() if group == val[0] and subgroup == val[1]])
                self.tree = {group:subgroup}
            elif group and not subgroup:
                self.book = {val[1]:set([key for key,val in book.items() if group == val[0]]) for val in book.values()}
                self.tree = {group:[val[1] for val in book.values() if group == val[0]]}
            elif not group and subgroup:
                self.book = set([key for key in book.keys()])
                self.tree = {val[0]:[val[1] for val in book.values() if group == val[0]] for val in book.values()}
            elif not group and not subgroup:
                self.book = book
                self.tree =  {val[0]:val[1] for val in book.values()}
        except Exception as e:
            print(e)
            print('Please consult self.allTree for all available groups and subgroups.')
