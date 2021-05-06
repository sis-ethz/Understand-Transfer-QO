
CREATE INDEX temp_tpch_idx_1 ON tpch_lineitem(l_partkey, l_quantity, l_shipmode, l_shipinstruct);
CREATE INDEX temp_tpch_idx_2 ON tpch_part(p_partkey, p_brand, p_container, p_size);

select
	sum(tpch_lineitem.l_extendedprice* (1 - tpch_lineitem.l_discount)) as revenue
from
	tpch_lineitem
	join tpch_part USE HASH(BUILD) on tpch_part.p_partkey = tpch_lineitem.l_partkey
where
	(tpch_lineitem.l_shipmode = 'AIR' or tpch_lineitem.l_shipmode = 'AIR REG')
	and tpch_lineitem.l_shipinstruct = 'DELIVER IN PERSON'
	and ((
		tpch_part.p_brand = 'Brand#12'
		and (tpch_part.p_container = 'SM CASE' or tpch_part.p_container = 'SM BOX' or tpch_part.p_container = 'SM PACK' or tpch_part.p_container = 'SM PKG')
		and tpch_lineitem.l_quantity >= 1 and tpch_lineitem.l_quantity <= 1 + 10
		and tpch_part.p_size between 1 and 5
	)
	or
	(
		tpch_part.p_brand = 'Brand#23'
		and (tpch_part.p_container = 'MED BAG' or tpch_part.p_container = 'MED BOX' or tpch_part.p_container = 'MED PKG' or tpch_part.p_container = 'MED PACK')
		and tpch_lineitem.l_quantity >= 10 and tpch_lineitem.l_quantity <= 10 + 10
		and tpch_part.p_size between 1 and 10
	)
	or
	(
		tpch_part.p_brand = 'Brand#34'
		and (tpch_part.p_container = 'LG CASE' or tpch_part.p_container = 'LG BOX' or tpch_part.p_container = 'LG PACK' or tpch_part.p_container = 'LG PKG')
		and tpch_lineitem.l_quantity >= 20 and tpch_lineitem.l_quantity <= 20 + 10
		and tpch_part.p_size between 1 and 15
	));


DROP INDEX tpch_lineitem.temp_tpch_idx_1;
DROP INDEX tpch_part.temp_tpch_idx_2;