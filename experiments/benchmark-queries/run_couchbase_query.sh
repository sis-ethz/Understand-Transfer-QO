# Run the original queries. 
# "preview" denotes the preview version of Couchbase with CBO 
python3 run_bench_query.py --system preview --engine Couchbase --data-set ssb --query-set couchbase_original --repeat 10
python3 run_bench_query.py --system preview --engine Couchbase --data-set tpch --query-set couchbase_original --repeat 10
python3 run_bench_query.py --system preview --engine Couchbase --data-set job_light --query-set couchbase_original --repeat 10

# Run the queries transformed from Postgres. 
# "original" denotes the original version of Couchbase without CBO 
python3 run_bench_query.py --system original --engine Couchbase --data-set ssb --query-set couchbase_trans_postgres --repeat 10
python3 run_bench_query.py --system original --engine Couchbase --data-set tpch --query-set couchbase_trans_postgres --repeat 10
python3 run_bench_query.py --system original --engine Couchbase --data-set job_light --query-set couchbase_trans_postgres --repeat 10

# Run the queries transformed from Mssql. The run times are bassicly the same as postgres except 2 queries 
# since they predict the same plan (as shown in the paper). 
# "original" denotes the original version of Couchbase without CBO 
python3 run_bench_query.py --system original --engine Couchbase --data-set ssb --query-set couchbase_trans_mssql --repeat 10
python3 run_bench_query.py --system original --engine Couchbase --data-set tpch --query-set couchbase_trans_mssql --repeat 10
python3 run_bench_query.py --system original --engine Couchbase --data-set job_light --query-set couchbase_trans_mssql --repeat 10
