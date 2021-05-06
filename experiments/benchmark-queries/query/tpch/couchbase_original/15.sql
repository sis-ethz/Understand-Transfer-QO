with revenue as (
	select
		tpch_lineitem.l_suppkey as supplier_no,
		sum(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)) as total_revenue
	from
		tpch_lineitem
	where
		tpch_lineitem.l_shipdate >= '1996-01-01'
		and tpch_lineitem.l_shipdate < '1996-04-01'
	group by
		tpch_lineitem.l_suppkey
)

select
	tpch_supplier.s_suppkey,
	tpch_supplier.s_name,
	tpch_supplier.s_address,
	tpch_supplier.s_phone,
	revenue.total_revenue
from
	tpch_supplier
	join revenue on tpch_supplier.s_suppkey = revenue.supplier_no
where
	revenue.total_revenue = (
		select
			RAW max(temp.total_revenue)
		from
			(
				select
					l0.l_suppkey as supplier_no,
					sum(l0.l_extendedprice * (1 - l0.l_discount)) as total_revenue
				from
					tpch_lineitem as l0
				where
					l0.l_shipdate >= '1996-01-01'
					and l0.l_shipdate < '1996-04-01'
				group by
					l0.l_suppkey
			) as temp
	)[0]
order by
	tpch_supplier.s_suppkey;
