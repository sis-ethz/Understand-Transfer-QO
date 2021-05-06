-- SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
-- FROM customer, lineorder, supplier, ddate   
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND c_nation = 'UNITED STATES'
--   AND s_nation = 'UNITED STATES'
--   AND d_year >= 1992 AND d_year <= 1997
-- GROUP BY c_city, s_city, d_year
-- ORDER BY d_year ASC, revenue DESC;

CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_custkey, lo_orderdate, lo_suppkey);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_year);
CREATE INDEX temp_ix_3 ON ssb_customer(c_custkey, c_nation);
CREATE INDEX temp_ix_4 ON ssb_supplier(s_suppkey, s_nation);

UPDATE STATISTICS FOR ssb_lineorder(lo_custkey, lo_orderdate, lo_suppkey);
UPDATE STATISTICS FOR ssb_ddate(d_datekey, d_year);
UPDATE STATISTICS FOR ssb_customer(c_custkey, c_nation);
UPDATE STATISTICS FOR ssb_supplier(s_suppkey, s_nation);

SELECT ssb_customer.c_city, ssb_supplier.s_city, ssb_ddate.d_year, SUM(ssb_lineorder.lo_revenue) AS revenue
FROM ssb_customer INNER JOIN ssb_lineorder ON ssb_lineorder.lo_custkey = ssb_customer.c_custkey
INNER JOIN ssb_supplier ON ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey
INNER JOIN ssb_ddate ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
WHERE ssb_customer.c_nation = 'UNITED STATES'
  AND ssb_supplier.s_nation = 'UNITED STATES'
  AND ssb_ddate.d_year >= 1992 AND ssb_ddate.d_year <= 1997
GROUP BY ssb_customer.c_city, ssb_supplier.s_city, ssb_ddate.d_year
ORDER BY ssb_ddate.d_year ASC, revenue DESC;

DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;
DROP INDEX ssb_customer.temp_ix_3;
DROP INDEX ssb_supplier.temp_ix_4;