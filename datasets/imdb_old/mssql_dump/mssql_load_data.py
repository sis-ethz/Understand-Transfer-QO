# mssql load data

import os
from tqdm import tqdm

total_file_num = 1484


os.system('/opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P "SQLServer123" -i /mnt/interpretable-cost-model/datasets/imdb_old/sqlserver_create_table_and_copy_data.sql -o /mnt/interpretable-cost-model/datasets/imdb_old/sql_server_out.txt')

for i in tqdm(range(total_file_num+1)):

    os.system(
        f'/opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P "SQLServer123" -i /mnt/interpretable-cost-model/datasets/imdb_old/mssql_dump/imdb_mssql_dump_{i}.sql -o /mnt/interpretable-cost-model/datasets/imdb_old/mssql_dump/imdb_mssql_dump_out_{i}.sql')

    # if i > 2:
    #     exit(1)
