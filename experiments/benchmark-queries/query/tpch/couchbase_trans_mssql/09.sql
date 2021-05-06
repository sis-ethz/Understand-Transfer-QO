CREATE INDEX temp_tpch_idx_1 ON tpch_lineitem(l_partkey, l_suppkey, l_orderkey);
CREATE INDEX temp_tpch_idx_2 ON tpch_part(p_partkey, p_name);
CREATE INDEX temp_tpch_idx_3 ON tpch_supplier(s_suppkey, s_nationkey);
CREATE INDEX temp_tpch_idx_4 ON tpch_partsupp(ps_suppkey, ps_partkey);
CREATE INDEX temp_tpch_idx_5 ON tpch_nation(n_nationkey);
CREATE INDEX temp_tpch_idx_6 ON tpch_orders(o_orderkey);



select
	profit.nation,
	profit.o1.o_year,
	sum(profit.amount) as sum_profit
from
	(
		select
			n1.n_name as nation,
			l1.l_extendedprice * (1 - l1.l_discount) - ps1.ps_supplycost * l1.l_quantity as amount
		from
			tpch_lineitem as l1
			join tpch_part as p1 USE HASH(BUILD)  on p1.p_partkey = l1.l_partkey
			join tpch_supplier as s1 USE HASH(BUILD)  on s1.s_suppkey = l1.l_suppkey
			join tpch_partsupp as ps1 USE HASH(BUILD)  on (ps1.ps_suppkey = l1.l_suppkey and ps1.ps_partkey = l1.l_partkey)
			join tpch_orders  as o1 USE HASH(BUILD) on o1.o_orderkey = l1.l_orderkey
			join tpch_nation  as n1 USE HASH(BUILD) on s1.s_nationkey = n1.n_nationkey
		where p1.p_name like '%green%'
	) as profit
group by
	profit.nation,
	profit.o1.o_year
order by
	profit.nation,
	profit.o1.o_year desc;

DROP INDEX tpch_lineitem.temp_tpch_idx_1;
DROP INDEX tpch_part.temp_tpch_idx_2;
DROP INDEX tpch_supplier.temp_tpch_idx_3;
DROP INDEX tpch_partsupp.temp_tpch_idx_4;
DROP INDEX tpch_nation.temp_tpch_idx_5;
DROP INDEX tpch_orders.temp_tpch_idx_6;
