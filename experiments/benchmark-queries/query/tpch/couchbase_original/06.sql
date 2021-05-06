select
	sum(tpch_lineitem.l_extendedprice * tpch_lineitem.l_discount) as revenue
from
	tpch_lineitem
where
	tpch_lineitem.l_shipdate >= '1994-01-01'
	and tpch_lineitem.l_shipdate < '1995-01-01'
	and tpch_lineitem.l_discount between 0.05 and 0.07
	and tpch_lineitem.l_quantity < 24;
