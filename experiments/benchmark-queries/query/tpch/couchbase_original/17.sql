select
	sum(tpch_lineitem.l_extendedprice) / 7.0 as avg_yearly
from
	tpch_lineitem
	join tpch_part on tpch_part.p_partkey = tpch_lineitem.l_partkey
where
	tpch_part.p_brand = 'Brand#23'
	and tpch_part.p_container = 'MED BOX'
	and tpch_lineitem.l_quantity < (
		select
			RAW 0.2 * avg(l0.l_quantity)
		from
			tpch_lineitem as l0
		where
			l0.l_partkey = tpch_part.p_partkey
	)[0];
