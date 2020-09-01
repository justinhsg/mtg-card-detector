import cv2
import numpy as np
import requests

import pickle
from os import path, mkdir
from helper_cv2 import phash

def cache_exists(set_code):
    if(path.isdir("./cache")):
        return (path.isfile(f"./cache/{set_code}_hash.npz") and path.isfile(f"./cache/{set_code}_name_map.dat"))
    return False
            
def im_from_url(url):
    raw_response = requests.get(url, stream=True).raw
    raw_image = np.asarray(bytearray(raw_response.read()), dtype='uint8')
    image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
    return image

def create_cache(set_code):
    if(not cache_exists(set_code)):
        set_url = "https://api.scryfall.com/sets/{}".format(set_code)
        response = requests.get(set_url).json()
        if(response['object'] == 'error'):
            print("Error code {} - {}".format(response['status'], response['details']), file = sys.stderr)
            return
        print("Creating cache for {}".format(response['name']))
        
        card_query_parts = response['search_uri'].split('&unique=prints')
        #remove basics
        card_query = "+-t%3Abasic&unique=prints".join(card_query_parts)
        
        response = requests.get(card_query).json()
        if(response['object'] == 'error'):
            print("Error code {} - {}".format(response['status'], response['details']), file = sys.stderr)
            return
        
        cards = []
        while(True):
            card_list = response['data']
            for card in card_list:
                cards.append({'name': card['name'], 'image_url': card['image_uris']['normal'], 'id': card['collector_number']})
            if(response['has_more']):
                response = requests.get(response['next_page']).json()
            else:
                break
        construct_cache(set_code, cards)
    return

def construct_cache(set_code, cards):
    n_cards = len(cards)
    print("{} unique cards found, excluding basic lands".format(n_cards))
    
    name_map = dict()
    names = list()
    hashes = np.zeros((len(cards),32*32*3), dtype=np.bool)
    for idx, card in enumerate(cards):
        print("Processed {}/{} cards...".format(idx, n_cards), end="\r")
        name = card['name']
        if(name not in name_map):
            name_map[name] = []
        name_map[name].append(idx)
        
        names.append(name)
        
        url = card['image_url']
        image = im_from_url(url)
        mod_image = image
        hashes[idx] = phash(mod_image)
    print("All {} cards processed. Now saving.".format(n_cards))
    
    if(not path.isdir("./cache")):
        print("Creating directory ./cache/")
        mkdir("./cache")
    
    with open("./cache/{}_name_map.dat".format(set_code), "wb") as outfile:
        pickle.dump(name_map, outfile, protocol = pickle.HIGHEST_PROTOCOL)
    
    
    names = np.array(names)
    np.savez("./cache/{}_hash".format(set_code), names=names, hashes=hashes)
    print("Saved in ./cache/{}_hash.npz and ./cache/{}_name_map.dat".format(set_code, set_code))
    

def import_decklist(filename, deck_name):
    if(not path.isfile(filename)):
        print(f"No such filename {filename}")
        return
    raw_list = ""
    with open(filename, "r") as infile:
        raw_list = infile.read()
    lines = raw_list.rstrip().split("\n")
    card_queries = []
    cards = []
    for line in lines:
        card_name = " ".join(line.split(" ")[1:])
        card_queries.append(f"%21%22{requests.utils.quote(card_name, safe='')}%22")
        #print(card_queries)
        #card_queries.append("%20{card_name}%20")
        if(len(card_queries)==25):
            scryfall_query = f"https://api.scryfall.com/cards/search?unique=art&q=%28{'+or+'.join(card_queries)}%29-t%3Dbasic"
            
            card_queries = []
            
            response = requests.get(scryfall_query).json()
            
            if(response['object'] == 'error'):
                print("Error code {} - {}".format(response['status'], response['details']), file = sys.stderr)
                return
                
            while(True):
                card_list = response['data']
                for card in card_list:
                    cards.append({'name': card['name'], 'image_url': card['image_uris']['normal'], 'id': card['collector_number']})
                if(response['has_more']):
                    response = requests.get(response['next_page']).json()
                else:
                    break
        
    if(len(card_queries) != 0):
        scryfall_query = f"https://api.scryfall.com/cards/search?unique=art&q=%28{'+or+'.join(card_queries)}%29-t%3Dbasic"
            
        card_queries = []
        
        response = requests.get(scryfall_query).json()
        
        if(response['object'] == 'error'):
            print("Error code {} - {}".format(response['status'], response['details']), file = sys.stderr)
            return
            
        while(True):
            card_list = response['data']
            for card in card_list:
                cards.append({'name': card['name'], 'image_url': card['image_uris']['normal'], 'id': card['collector_number']})
            if(response['has_more']):
                response = requests.get(response['next_page']).json()
            else:
                break
    
    construct_cache(deck_name ,cards)
    return
    
    
def load_cache(set_code):
    if cache_exists(set_code):
    
    
    
        d = None
        with open(f"./cache/{set_code}_name_map.dat", "rb") as infile:
            d = pickle.load(infile)
            
        
        hashes = np.load(f"./cache/{set_code}_hash.npz")
        
        return (d, hashes['names'], hashes['hashes'])
    else:
        return None
    
if __name__ == "__main__":
    import sys
    if(len(sys.argv) != 3):
        print("Usage: python create-cache.py <set-code>", file = sys.stderr)
        sys.exit(1)
    else:
        #create_cache(sys.argv[1])
        print(import_decklist(sys.argv[1], sys.argv[2]))
        #construct_cache()
        #load_cache()
        #print(cache_exists(sys.argv[1]))
        