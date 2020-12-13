

table_list = ['customer', 'lineitem', 'nation',
              'orders', 'part', 'partsupp', 'region', 'supplier']


for t in table_list:
    with open(t + '.tbl', 'r') as f, open(t + '_fixed.tbl', 'w') as g:
        line = f.readline()
        while line:
            line = line.replace("\n", '').strip('|')
            g.write(line + '\n')
            line = f.readline()
