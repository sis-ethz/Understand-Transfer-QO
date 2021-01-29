import os
import json


def postgres_to_couchbase(from_file, to_file):
    short_for_table = {
        'l': 'lineitem',
        's': 'supplier',
        'n': 'nation',
        'p': 'part',
        'r': 'region',
        'ps': 'partsupp',
        'c': 'customer',
        'o': 'order'
    }
    
    query = ''
    with open(from_file, 'r') as f:
        query = f.read()

    for short in short_for_table.keys():
        if short + '_' in query:
            query = query.replace(short + '_', 'tpch_' + short_for_table[short] + '.' + short + '_')

    with open(to_file, 'w') as f:
        f.write(query)

    


if __name__ == "__main__":
    from_folder = './original/'
    to_folder = './original/'

    files = os.listdir(from_folder)
    for f in files:
        postgres_to_couchbase(os.path.join(from_folder, f), os.path.join(to_folder, f))

