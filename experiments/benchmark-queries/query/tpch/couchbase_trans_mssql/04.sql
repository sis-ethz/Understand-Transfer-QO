select
	tpch_orders.o_orderpriority,
	count(*) as order_count
from
	tpch_orders
where
	tpch_orders.o_orderdate >= '1993-07-01'
	and tpch_orders.o_orderdate < '1993-10-01'
	and exists (
		select
			*
		from
			tpch_lineitem as l0
		where
			l0.l_orderkey = tpch_orders.o_orderkey
			and l0.l_commitdate < l0.l_receiptdate
	)
group by
	tpch_orders.o_orderpriority
order by
	tpch_orders.o_orderpriority;
