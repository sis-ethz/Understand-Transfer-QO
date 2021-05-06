select
	tpch_supplier.s_name,
	tpch_supplier.s_address
from
	tpch_supplier
	join tpch_nation on tpch_supplier.s_nationkey = tpch_nation.n_nationkey
where
	tpch_supplier.s_suppkey in (
		select
			RAW ps0.ps_suppkey
		from
			tpch_partsupp as ps0
		where
			ps0.s_partkey in (
				select
					RAW p0.p_partkey
				from
					tpch_part as p0
				where
					p0.p_name like 'forest%'
			)
			and ps0.s_availqty > (
				select
					RAW 0.5 * sum(l0.l_quantity)
				from
					tpch_lineitem as l0
				where
					l0.l_partkey = ps0.s_partkey
					and l0.l_suppkey = ps0.s_suppkey
					and l0.l_shipdate >= '1994-01-01'
					and l0.l_shipdate < '1995-01-01'
			)[0]
	)
	and tpch_nation.n_name = 'CANADA'
order by
	tpch_supplier.s_name;
