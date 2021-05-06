select
	custsale.cntrycode,
	count(*) as numcust,
	sum(custsale.acctbal) as totacctbal
from
	(
		select
			SUBSTR(c1.c_phone, 1, 2) as cntrycode,
			c1.c_acctbal as acctbal
		from
			tpch_customer as c1
		where
			SUBSTR(c1.c_phone, 1, 2) in
				['13', '31', '23', '29', '30', '18', '17']
			and c1.c_acctbal > (
				select
					avg(c2.c_acctbal)
				from
					tpch_customer as c2
				where
					c2.c_acctbal > 0.00
					and SUBSTR(c2.c_phone, 1, 2) in
						['13', '31', '23', '29', '30', '18', '17']
			)
			and not exists (
				select
					*
				from
					tpch_orders as o1
				where
					o1.o_custkey = c1.c_custkey
			)
	) as custsale
group by
	custsale.cntrycode
order by
	custsale.cntrycode;

