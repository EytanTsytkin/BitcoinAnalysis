import cloudscraper
import bs4
import json

#https://checkbitcoinaddress.com/
headers = {
        'User-Agent': 'Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10',
        'referer': 'https://blockchain.com/',
    }
Typedict = {'services/others': ['checkbitcoinaddress']}

def getAddresses_cloud(pageNum):
    addresses = []
    scraper = cloudscraper.create_scraper()
    wallet = scraper.get(f'https://checkbitcoinaddress.com/submitted-links?page={pageNum}').content
    soup = bs4.BeautifulSoup(wallet,'html.parser')
    table_html = soup.find('table',class_='table table-hover')
    href_html = table_html.findAll('a')
    print(href_html)
    for one in href_html:
        if one.has_attr('target') == False:
            addresses.append(one.get_text())
    return addresses


def makeAddressBook():
    address_list = []
    AddressBook=dict()
    for i in range(1,155):
        onepage_address = getAddresses_cloud(i)
        AddressBook.extend(onepage_address)

    for key in Typedict.keys():
        for name in Typedict[key]:
            for addr in address_list:
                AddressBook[addr] = (key, name)

    with open("..\JsonFile\AddressBook.json",'w') as i:
            json.dump(AddressBook,i)
    return AddressBook

makeAddressBook()