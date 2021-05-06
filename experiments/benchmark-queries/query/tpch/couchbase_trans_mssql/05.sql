CREATE INDEX temp_tpch_idx_1 ON tpch_customer(c_custkey);
CREATE INDEX temp_tpch_idx_2 ON tpch_orders(o_custkey, o_orderkey, o_orderdate);
CREATE INDEX temp_tpch_idx_3 ON tpch_lineitem(l_orderkey, l_suppkey);
CREATE INDEX temp_tpch_idx_4 ON tpch_supplier(s_suppkey, s_nationkey);
CREATE INDEX temp_tpch_idx_5 ON tpch_nation(n_nationkey, n_regionkey);
CREATE INDEX temp_tpch_idx_6 ON tpch_region(r_regionkey, r_name);

SELECT tpch_nation.n_name,
       SUM(tpch_lineitem.l_extendedprice * (1 - tpch_lineitem.l_discount)) AS revenue
FROM tpch_customer
    JOIN tpch_orders USE HASH(BUILD) ON tpch_customer.c_custkey = tpch_orders.o_custkey
    JOIN tpch_lineitem USE HASH(BUILD) ON tpch_lineitem.l_orderkey = tpch_orders.o_orderkey
    JOIN tpch_supplier  USE HASH(BUILD)  ON tpch_lineitem.l_suppkey = tpch_supplier.s_suppkey
    JOIN tpch_nation  USE HASH(BUILD) ON tpch_supplier.s_nationkey = tpch_nation.n_nationkey
    JOIN tpch_region USE HASH(BUILD)  ON tpch_nation.n_regionkey = tpch_region.r_regionkey
WHERE tpch_region.r_name = 'ASIA'
    AND tpch_orders.o_orderdate >= '1994-01-01'
    AND tpch_orders.o_orderdate < '1995-01-01'
GROUP BY tpch_nation.n_name
ORDER BY revenue DESC ;


DROP INDEX tpch_customer.temp_tpch_idx_1;
DROP INDEX tpch_orders.temp_tpch_idx_2;
DROP INDEX tpch_lineitem.temp_tpch_idx_3;
DROP INDEX tpch_supplier.temp_tpch_idx_4;
DROP INDEX tpch_nation.temp_tpch_idx_5;
DROP INDEX tpch_region.temp_tpch_idx_6;