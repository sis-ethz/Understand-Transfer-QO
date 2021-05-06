select
	c1.c_count,
	count(*) as custdist
from
	(
		select
			tpch_customer.c_custkey as c_custkey,
			count(tpch_orders.o_orderkey) as c_count
		from
			tpch_customer left outer join tpch_orders on
				tpch_customer.c_custkey = tpch_orders.o_custkey
				and tpch_orders.o_comment not like '%special%requests%'
		group by
			tpch_customer.c_custkey
	) as c1
group by
	c1.c_count
order by
	custdist desc,
	c1.c_count desc;
