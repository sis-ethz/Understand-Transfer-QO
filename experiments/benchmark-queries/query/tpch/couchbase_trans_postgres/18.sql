CREATE INDEX temp_tpch_idx_1 ON tpch_customer(c_custkey);
CREATE INDEX temp_tpch_idx_2 ON tpch_orders(o_custkey);
CREATE INDEX temp_tpch_idx_3 ON tpch_lineitem(l_orderkey);
CREATE INDEX temp_tpch_idx_4 ON tpch_lineitem(l_quantity);

select
	tpch_customer.c_name,
	tpch_customer.c_custkey,
	tpch_orders.o_orderkey,
	tpch_orders.o_orderdate,
	tpch_orders.o_totalprice,
	sum(tpch_lineitem.l_quantity)
from
	tpch_customer
	join tpch_orders USE HASH(BUILD) on tpch_customer.c_custkey = tpch_orders.o_custkey
	join tpch_lineitem USE NL on tpch_orders.o_orderkey = tpch_lineitem.l_orderkey
where
	tpch_orders.o_orderkey in (
		select
			RAW l0.l_orderkey
		from
			tpch_lineitem as l0
		group by
			l0.l_orderkey having
				sum(l0.l_quantity) > 300
	)
group by
	tpch_customer.c_name,
	tpch_customer.c_custkey,
	tpch_orders.o_orderkey,
	tpch_orders.o_orderdate,
	tpch_orders.o_totalprice
order by
	tpch_orders.o_totalprice desc,
	tpch_orders.o_orderdate
limit 100;


DROP INDEX tpch_customer.temp_tpch_idx_1;
DROP INDEX tpch_orders.temp_tpch_idx_2;
DROP INDEX tpch_lineitem.temp_tpch_idx_3;
DROP INDEX tpch_lineitem.temp_tpch_idx_4;

