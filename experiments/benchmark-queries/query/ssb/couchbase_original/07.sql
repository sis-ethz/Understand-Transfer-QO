-- SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
-- FROM customer, lineorder, supplier, ddate   
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND c_region = 'ASIA'
--   AND s_region = 'ASIA'
--   AND d_year >= 1992 AND d_year <= 1997
-- GROUP BY c_nation, s_nation, d_year
-- ORDER BY d_year ASC, revenue DESC;
CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_custkey, lo_orderdate, lo_suppkey);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_year);
CREATE INDEX temp_ix_3 ON ssb_customer(c_custkey, c_region);
CREATE INDEX temp_ix_4 ON ssb_supplier(s_suppkey, s_region);

UPDATE STATISTICS FOR ssb_lineorder(lo_custkey, lo_orderdate, lo_suppkey);
UPDATE STATISTICS FOR ssb_ddate(d_datekey, d_year);
UPDATE STATISTICS FOR ssb_customer(c_custkey, c_region);
UPDATE STATISTICS FOR ssb_supplier(s_suppkey, s_region);

SELECT ssb_customer.c_nation, ssb_supplier.s_nation, ssb_ddate.d_year, SUM(ssb_lineorder.lo_revenue) AS revenue
FROM ssb_customer 
INNER JOIN ssb_lineorder ON ssb_lineorder.lo_custkey = ssb_customer.c_custkey
INNER JOIN ssb_supplier ON ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey
INNER JOIN ssb_ddate ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
WHERE 
 ssb_customer.c_region = 'ASIA'
  AND ssb_supplier.s_region = 'ASIA'
  AND ssb_ddate.d_year >= 1992 AND ssb_ddate.d_year <= 1997
GROUP BY ssb_customer.c_nation, ssb_supplier.s_nation, ssb_ddate.d_year
ORDER BY ssb_ddate.d_year ASC, revenue DESC;

DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;
DROP INDEX ssb_customer.temp_ix_3;
DROP INDEX ssb_supplier.temp_ix_4;

-- SELECT COUNT(*)  FROM customer WHERE c_region = 'ASIA'
-- AND c_custkey IN (
-- SELECT lo_custkey
-- FROM lineorder);

-- SELECT COUNT(*) FROM ddate where d_datekey in 
-- (SELECT lo_orderdate FROM customer, lineorder, supplier   
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_region = 'ASIA'
--   AND s_region = 'ASIA')
--  AND d_year >= 1992 AND d_year <= 1997;


-- SELECT COUNT(*) FROM ddate where d_datekey in 
-- (SELECT lo_orderdate
-- FROM customer, lineorder, supplier   
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_region = 'ASIA'
--   AND s_region = 'ASIA');

-- SELECT COUNT(*)  FROM customer WHERE c_custkey IN (
-- SELECT lo_custkey
-- FROM lineorder);

