select
	all_nations.o1.o_year,
	sum(case
		when all_nations.nation = 'BRAZIL' then volume
		else 0
	end) / sum(all_nations.volume) as mkt_share
from
	(
		select
			l1.l_extendedprice * (1 - l1.l_discount) as volume,
			n2.n_name as nation
		from
			tpch_lineitem AS l1
			JOIN tpch_supplier AS s1 USE HASH(BUILD)  ON s1.s_suppkey = l1.l_suppkey
			JOIN tpch_part AS p1 USE HASH(BUILD)  ON p1.p_partkey = l1.l_partkey
			JOIN tpch_orders AS o1 USE HASH(BUILD)  ON l1.l_orderkey = o1.o_orderkey
			JOIN tpch_customer AS c1 USE HASH(BUILD)  ON o1.o_custkey = c1.c_custkey
			JOIN tpch_nation AS n1 USE HASH(BUILD)  ON c1.c_nationkey = n1.n_nationkey
			JOIN tpch_nation AS n2 USE HASH(BUILD)  ON s1.s_nationkey = n2.n_nationkey
			JOIN tpch_region AS r1 USE HASH(BUILD) ON n1.n_regionkey = r1.r_regionkey
		where
			r1.r_name = 'AMERICA'
			and o1.o_orderdate between '1995-01-01' and '1996-12-31'
			and p1.p_type = 'ECONOMY ANODIZED STEEL'
	) as all_nations
group by
	all_nations.o1.o_year
order by
	all_nations.o1.o_year;
