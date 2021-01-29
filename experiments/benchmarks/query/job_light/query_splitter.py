
import os


def split(file_dir, file_name):
    with open(os.path.join(file_dir, file_name)) as f:
        queries = f.read()

    queries = queries.split('\n')
    for i in range(len(queries)):
        with open(os.path.join(file_dir, str(i+1) + '.sql'), 'w') as f:
            f.write(queries[i]) 

if __name__ == '__main__':
    file_loc = './original/'
    file_name = 'queries.sql'
    split(file_loc, file_name)