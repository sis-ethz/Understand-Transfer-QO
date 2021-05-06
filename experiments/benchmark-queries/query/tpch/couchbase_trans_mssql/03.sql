CREATE INDEX temp_tpch_idx_1 ON tpch_customer(c_custkey, c_mktsegment);
CREATE INDEX temp_tpch_idx_2 ON tpch_orders(o_custkey, o_orderkey, o_orderdate);
CREATE INDEX temp_tpch_idx_2 ON tpch_lineitem(l_orderkey, l_shipdate);


select
	tpch_lineitem.l_orderkey,
	sum(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)) as revenue,
	tpch_orders.o_orderdate,
	tpch_orders.o_shippriority
from
	tpch_customer
	JOIN tpch_orders USE HASH(PROBE) ON tpch_customer.c_custkey = tpch_orders.o_custkey
	JOIN tpch_lineitem USE HASH(PROBE) ON tpch_lineitem.l_orderkey = tpch_orders.o_orderkey
where
	tpch_customer.c_mktsegment = 'BUILDING'
	and tpch_orders.o_orderdate < '1995-03-15'
	and tpch_lineitem.l_shipdate > '1995-03-15'
group by
	tpch_lineitem.l_orderkey,
	tpch_orders.o_orderdate,
	tpch_orders.o_shippriority
order by
	revenue desc,
	tpch_orders.o_orderdate
limit 10
;

DROP INDEX tpch_customer.temp_tpch_idx_1;
DROP INDEX tpch_orders.temp_tpch_idx_2;
DROP INDEX tpch_lineitem.temp_tpch_idx_3;