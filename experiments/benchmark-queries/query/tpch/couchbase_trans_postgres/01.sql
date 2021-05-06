select
	tpch_lineitem.l_returnflag,
	tpch_lineitem.l_linestatus,
	sum(tpch_lineitem.l_quantity) as sum_qty,
	sum(tpch_lineitem.l_extendedprice) as sum_base_price,
	sum(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)) as sum_distpch_customer,
	sum(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount) * (1 + tpch_lineitem.l_tax)) as sum_charge,
	avg(tpch_lineitem.l_quantity) as avg_qty,
	avg(tpch_lineitem.l_extendedprice) as avg_price,
	avg(tpch_lineitem.l_discount) as avg_disc,
	count(*) as count_order
from
	tpch_lineitem
where
	tpch_lineitem.l_shipdate <= '1998-09-01'
group by
	tpch_lineitem.l_returnflag,
	tpch_lineitem.l_linestatus
order by
	tpch_lineitem.l_returnflag,
	tpch_lineitem.l_linestatus
;
