select
	shipping.supp_nation,
	shipping.cust_nation,
	shipping.l1.l_year,
	sum(shipping.volume) as revenue
from
	(
		select
			n1.n_name as supp_nation,
			n2.n_name as cust_nation,
			l1.l_extendedprice * (1 - l1.l_discount) as volume
		from
			tpch_supplier AS s1
			JOIN tpch_lineitem AS l1 USE NL  ON s1.s_suppkey = l1.l_suppkey
			JOIN tpch_orders AS o1 USE HASH(BUILD) ON o1.o_orderkey = l1.l_orderkey
			JOIN tpch_customer AS c1 USE HASH(BUILD) ON c1.c_custkey = o1.o_custkey
			JOIN tpch_nation AS n1 USE HASH(BUILD) ON s1.s_nationkey = n1.n_nationkey
			JOIN tpch_nation AS n2 USE HASH(BUILD) ON c1.c_nationkey = n2.n_nationkey
		where
			(
				(n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
				or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
			)
			and l1.l_shipdate between '1995-01-01' and '1996-12-31'
	) as shipping
group by
	shipping.supp_nation,
	shipping.cust_nation,
	shipping.l1.l_year
order by
	shipping.supp_nation,
	shipping.cust_nation,
	shipping.l1.l_year;
