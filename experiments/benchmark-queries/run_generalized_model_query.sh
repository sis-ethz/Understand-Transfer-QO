# Run all original queries
python3 run_bench_query.py --engine Postgres --data-set ssb --query-set postgres_original --repeat 10
python3 run_bench_query.py --engine Postgres --data-set tpch --query-set postgres_original --repeat 10
python3 run_bench_query.py --engine Postgres --data-set job_light --query-set postgres_original --repeat 10

# Run all queries with model predicted hints
python3 run_bench_query.py --engine Postgres --data-set ssb --query-set postgres_model --repeat 10
python3 run_bench_query.py --engine Postgres --data-set tpch --query-set postgres_model --repeat 10
python3 run_bench_query.py --engine Postgres --data-set job_light --query-set postgres_model --repeat 10