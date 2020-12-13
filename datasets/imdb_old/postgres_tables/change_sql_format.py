import os

sql_file_name = 'imdb_postgres_dump.sql'
sql_out_file_name = 'imdb_mssql_dump_{}.sql'
out_file_dir = '../mssql_dump/'

line_cnt = 0
file_cnt = 0
with open(sql_file_name, encoding='utf8') as f:

    # g = open(os.path.join(out_file_dir, sql_out_file_name.format(
    #     file_cnt)), 'w', encoding='utf8')

    g = open(os.path.join(out_file_dir, sql_out_file_name.format(
        'person_info')), 'w', encoding='utf8')

    g.write("USE imdb\nGO\n")

    line = f.readline()

    while line:
        if 'set' in line.lower():
            line = '-- ' + line

        line = line.replace('\n', '').replace('public.', '')

        if 'INSERT INTO person_info' not in line:
            line = f.readline()
            continue

        g.write(line + '\n')

        if line_cnt % 50000 == 0:
            g.write('GO\n')
        #     file_cnt += 1
        #     g.close()
        #     g = open(os.path.join(out_file_dir, sql_out_file_name.format(
        #         file_cnt)), 'w', encoding='utf8')
        #     g.write("USE imdb\nGO\n")

        line = f.readline()
        line_cnt += 1

    g.write('GO\n')
    g.close()

print("file cnt = ", file_cnt)
