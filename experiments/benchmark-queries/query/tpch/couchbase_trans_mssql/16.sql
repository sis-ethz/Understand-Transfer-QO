CREATE INDEX temp_tpch_idx_1 ON tpch_partsupp(ps_partkey, ps_suppkey);
CREATE INDEX temp_tpch_idx_2 ON tpch_part(p_partkey, p_brand, p_type, p_size);
CREATE INDEX temp_tpch_idx_3 ON tpch_supplier(s_comment);


select
	tpch_part.p_brand,
	tpch_part.p_type,
	tpch_part.p_size,
	count(distinct tpch_partsupp.ps_suppkey) as supplier_cnt
from
	tpch_partsupp
	join tpch_part USE HASH(BUILD) on tpch_part.p_partkey = tpch_partsupp.ps_partkey
where
	tpch_part.p_brand <> 'Brand#45'
	and tpch_part.p_type not like 'MEDIUM POLISHED%'
	and tpch_part.p_size in [49, 14, 23, 45, 19, 3, 36, 9]
	and tpch_partsupp.ps_suppkey not in (
		select
			RAW s0.s_suppkey
		from
			tpch_supplier as s0
		where
			s0.s_comment like '%Customer%Complaints%'
	)
group by
	tpch_part.p_brand,
	tpch_part.p_type,
	tpch_part.p_size
order by
	supplier_cnt desc,
	tpch_part.p_brand,
	tpch_part.p_type,
	tpch_part.p_size;


DROP INDEX tpch_partsupp.temp_tpch_idx_1;
DROP INDEX tpch_part.temp_tpch_idx_2;
DROP INDEX tpch_supplier.temp_tpch_idx_3;