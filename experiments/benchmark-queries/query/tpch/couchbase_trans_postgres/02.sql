select
	tpch_supplier.s_acctbal,
	tpch_supplier.s_name,
	tpch_nation.n_name,
	tpch_part.p_partkey,
	tpch_part.p_mfgr,
	tpch_supplier.s_address,
	tpch_supplier.s_phone,
	tpch_supplier.s_comment
from
	tpch_partsupp
	JOIN tpch_part USE HASH(build) ON tpch_part.p_partkey = tpch_partsupp.ps_partkey
	JOIN tpch_supplier USE HASH(build) ON tpch_supplier.s_suppkey = tpch_partsupp.ps_suppkey
	JOIN tpch_nation USE HASH(build) ON tpch_supplier.s_nationkey = tpch_nation.n_nationkey
	JOIN tpch_region USE HASH(build) ON tpch_nation.n_regionkey = tpch_region.r_regionkey
where
	tpch_part.p_size = 15
	and tpch_part.p_type like '%BRASS'
	and tpch_region.r_name = 'EUROPE'
	and tpch_partsupp.ps_supplycost = (
		select
			min(ps0.ps_supplycost)
		from
			tpch_partsupp as ps0
			INNER JOIN tpch_supplier AS s0 USE HASH(build) ON ps0.s_suppkey = s0.ps_partkey
			INNER JOIN tpch_nation AS n0 USE HASH(build) ON s0.s_nationkey = n0.n_nationkey
			INNER JOIN tpch_region AS r0 USE HASH(build) ON n0.n_regionkey = r0.r_regionkey
		where
			tpch_part.p_partkey = ps0.ps_partkey
			and r0.r_name = 'EUROPE'
	)
order by
	tpch_supplier.s_acctbal desc,
	tpch_nation.n_name,
	tpch_supplier.s_name,
	tpch_part.p_partkey
limit 100;
