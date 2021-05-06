CREATE INDEX temp_tpch_idx_1 ON tpch_lineitem(l_orderkey, l_returnflag);
CREATE INDEX temp_tpch_idx_2 ON tpch_orders(o_orderkey, o_custkey);
CREATE INDEX temp_tpch_idx_3 ON tpch_customer(c_custkey, c_nationkey);
CREATE INDEX temp_tpch_idx_4 ON tpch_nation(n_nationkey);

UPDATE STATISTICS FOR tpch_lineitem(l_orderkey, l_returnflag);
UPDATE STATISTICS FOR tpch_orders(o_orderkey, o_custkey);
UPDATE STATISTICS FOR tpch_customer(c_custkey, c_nationkey);
UPDATE STATISTICS FOR tpch_nation(n_nationkey);


select
	tpch_customer.c_custkey,
	tpch_customer.c_name,
	sum(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)) as revenue,
	tpch_customer.c_acctbal,
	tpch_nation.n_name,
	tpch_customer.c_address,
	tpch_customer.c_phone,
	tpch_customer.c_comment
from
	tpch_lineitem
	join tpch_orders on tpch_lineitem.l_orderkey = tpch_orders.o_orderkey
	join tpch_customer on tpch_customer.c_custkey = tpch_orders.o_custkey
	join tpch_nation on tpch_customer.c_nationkey = tpch_nation.n_nationkey
where
	tpch_orders.o_orderdate >=  '1993-10-01'
	and tpch_orders.o_orderdate <  '1994-01-01'
	and tpch_lineitem.l_returnflag = 'R'
group by
	tpch_customer.c_custkey,
	tpch_customer.c_name,
	tpch_customer.c_acctbal,
	tpch_customer.c_phone,
	tpch_nation.n_name,
	tpch_customer.c_address,
	tpch_customer.c_comment
order by
	revenue desc
limit 20
;

DROP INDEX tpch_lineitem.temp_tpch_idx_1;
DROP INDEX tpch_orders.temp_tpch_idx_2;
DROP INDEX tpch_customer.temp_tpch_idx_3;
DROP INDEX tpch_nation.temp_tpch_idx_4;
