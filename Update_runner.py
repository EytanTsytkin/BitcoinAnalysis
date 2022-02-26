import os
import sys
import time
import PATHS
from AddressBook import AddressBook

def test_n_times_multi(n, start, stop):
    test_results = []
    for test in range(n):
        test_results.append(test_multi_update(start, stop))
    return test_results


def test_n_times(n, start, stop):
    test_results = []
    for test in range(n):
        test_results.append(test_update(start, stop))
    return test_results


def test_update(start, stop):
    ab = AddressBook()
    t = time.time()
    ab.update_range(ab.address_book.keys(), start=start, stop=stop)
    print(f'Total time for 100 blocks:{time.time() - t}')
    return time.time() - t


def test_multi_update(start, stop):
    ab = AddressBook()
    t = time.time()
    ab.update_range_multiproc(ab.address_book.keys(), start=start, stop=stop)
    print(f'Total time for 100 blocks:{time.time() - t}')
    return time.time() - t


def test_merge(test_set=None):
    ab = AddressBook()
    if not test_set:
        ab.update_addresses = set(random.sample(ab.address_book.keys(), 100))
        test_set = ab.update_addresses
    else:
        ab.update_addresses = test_set
    print('First run:')
    ab.merge_vectors()
    print('Second run:')
    ab.merge_vectors()
    print('Third run:')
    ab.merge_vectors()
    return test_set


def merge(addressbook):
    t = time.time()
    if not addressbook:
        addressbook = AddressBook()
    found_addresses = set([address.replace('.csv', '') for address in os.listdir(PATHS.ADDRESS_VECTORS_PATH)])
    addressbook.update_addresses = set(addressbook.address_book.keys()).difference(set(found_addresses))
    addressbook.merge_vectors()
    print(f'Total time for merging {len(addressbook.update_addresses)} blocks:{time.time() - t}')

def update(start, stop, merge=False):
    ab = AddressBook()
    t = time.time()
    ab.update_range(ab.address_book.keys(), start=start, stop=stop)
    print(f'Total time for reading {stop-start} blocks:{time.time() - t}')
    if merge:
        merge(ab)


if __name__ == '__main__':
    merge()