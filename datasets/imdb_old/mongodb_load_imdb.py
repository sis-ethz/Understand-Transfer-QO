from pymongo import *
import pandas as pd
import numpy as np

client = MongoClient('localhost', 27017)
db = client["imdb"]
db = client.imdb

print(f'db {db}')

columns = {
    'movie_info_idx': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'movie_info': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'movie_companies': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
    'movie_keyword': ['id', 'movie_id', 'keyword_id'],
    'cast_info': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
    'title': ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
              'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years',
              'md5sum']
}


# ===============================
for t in columns.keys():
    table = db[t]

    x = table.delete_many({})
    print(f"deleted {x.deleted_count}")

    data = pd.read_csv(f'tables/{t}.csv', sep=',',
                       header=None, error_bad_lines=False)

    print("read from csv: ", data)

    # data = data.drop(data.columns[-1], axis=1)
    # print(data)
    data.columns = columns[t]

    # Set up a dictionary before import
    data = [x[1].to_dict() for x in data.iterrows()]
    # Insert dictionary data on table
    table.insert_many(data)

    print("loaded ", table.count(), f" lines for {t}")
# ===============================
