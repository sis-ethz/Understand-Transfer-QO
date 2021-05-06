select
	s1.s_name,
	count(*) as numwait
from
	tpch_supplier as s1
	join tpch_lineitem as l1 on s1.s_suppkey = l1.l_suppkey
	join tpch_orders as o1 on o1.o_orderkey = l1.l_orderkey
	join tpch_nation as n1 on s1.s_nationkey = n1.n_nationkey
where
	o1.o_orderstatus = false
	and l1.l_receiptdate > l1.l_commitdate
	and exists (
		select
			*
		from
			tpch_lineitem l2
		where
			l2.l_orderkey = l1.l_orderkey
			and l2.l_suppkey <> l1.l_suppkey
	)
	and not exists (
		select
			*
		from
			lineitem l3
		where
			l3.l_orderkey = l1.l_orderkey
			and l3.l_suppkey <> l1.l_suppkey
			and l3.l_receiptdate > l3.l_commitdate
	)
	and n1.n_name = 'SAUDI ARABIA'
group by
	s1.s_name
order by
	numwait desc,
	s1.s_name
limit 100;

