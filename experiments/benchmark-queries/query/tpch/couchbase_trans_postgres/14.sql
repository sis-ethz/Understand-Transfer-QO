CREATE INDEX temp_tpch_idx_1 ON tpch_lineitem(l_partkey, l_shipdate);
CREATE INDEX temp_tpch_idx_2 ON tpch_part(p_partkey);


select
	100.00 * sum(case
		when tpch_part.p_type like 'PROMO%'
		then tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)
		else 0
	end) / sum(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)) as promo_revenue
from
	tpch_lineitem
	join tpch_part USE HASH(BUILD) on tpch_lineitem.l_partkey = tpch_part.p_partkey
where
	tpch_lineitem.l_shipdate >= '1995-09-01'
	and tpch_lineitem.l_shipdate < '1995-10-01'
;


DROP INDEX tpch_lineitem.temp_tpch_idx_1;
DROP INDEX tpch_part.temp_tpch_idx_2;