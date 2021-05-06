CREATE INDEX temp_tpch_idx_1 ON tpch_orders(o_orderkey);
CREATE INDEX temp_tpch_idx_2 ON tpch_lineitem(l_orderkey, l_shipmode, l_commitdate, l_receiptdate, l_shipdate);

select
	l1.l_shipmode,
	sum(case
		when o1.o_orderpriority = '1-URGENT' or o1.o_orderpriority = '2-HIGH'
		then 1
		else 0
	end) as high_line_count,
	sum(case
		when o1.o_orderpriority <> '1-URGENT' or o1.o_orderpriority <> '2-HIGH'
		then 1
		else 0
	end) as low_line_count
from
	tpch_orders as o1
	join tpch_lineitem as l1 USE HASH(BUILD) on o1.o_orderkey = l1.l_orderkey
where
	(l1.l_shipmode = 'MAIL' or l1.l_shipmode = 'SHIP')
	and l1.l_commitdate < l1.l_receiptdate
	and l1.l_shipdate < l1.l_commitdate
	and l1.l_receiptdate >= '1994-01-01'
	and l1.l_receiptdate < '1995-01-01'
group by
	l1.l_shipmode
order by
	l1.l_shipmode
;

DROP INDEX tpch_orders.temp_tpch_idx_1;
DROP INDEX tpch_lineitem.temp_tpch_idx_2;
