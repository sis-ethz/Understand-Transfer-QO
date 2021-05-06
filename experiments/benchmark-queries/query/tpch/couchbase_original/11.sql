CREATE INDEX temp_tpch_idx_1 ON tpch_partsupp(ps_suppkey);
CREATE INDEX temp_tpch_idx_2 ON tpch_supplier(s_suppkey, s_nationkey);
CREATE INDEX temp_tpch_idx_3 ON tpch_nation(n_nationkey, n_name);

UPDATE STATISTICS FOR tpch_partsupp(ps_suppkey);
UPDATE STATISTICS FOR tpch_supplier(s_suppkey, s_nationkey);
UPDATE STATISTICS FOR tpch_nation(n_nationkey, n_name);


select
	ps2.ps_partkey,
	sum(ps2.ps_supplycost * ps2.ps_availqty) as v1
from
	tpch_partsupp as ps2
	join tpch_supplier as s2 on ps2.ps_suppkey = s2.s_suppkey
	join tpch_nation as n2 on s2.s_nationkey = n2.n_nationkey
where n2.n_name = 'GERMANY'
group by
	ps2.ps_partkey 
having
	sum(ps2.ps_supplycost * ps2.ps_availqty) >
	(
		select
			sum(ps1.ps_supplycost * ps1.ps_availqty) * 0.00000100000000
		from
			tpch_partsupp as ps1 
			join tpch_supplier as s1 on ps1.ps_suppkey = s1.s_suppkey
			join tpch_nation as n1 on s1.s_nationkey = n1.n_nationkey
		where
			n1.n_name = 'GERMANY'
	) 
order by
	v1 desc;

DROP INDEX tpch_partsupp.temp_tpch_idx_1;
DROP INDEX tpch_supplier.temp_tpch_idx_2;
DROP INDEX tpch_nation.temp_tpch_idx_3;
