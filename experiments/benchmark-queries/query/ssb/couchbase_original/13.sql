-- SELECT ssb_ddate.d_year, ssb_supplier.s_city, ssb_part.p_brand1, sum(ssb_lineorder.lo_revenue-ssb_lineorder.lo_supplycost) AS profit
-- FROM ddate, customer, supplier, part, lineorder
-- WHERE ssb_lineorder.lo_custkey = ssb_customer.c_custkey
--   AND ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey
--   AND ssb_lineorder.lo_partkey = ssb_part.p_partkey
--   AND ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
--   AND ssb_customer.c_region = 'AMERICA'
--   AND ssb_supplier.s_nation = 'UNITED STATES'
--   AND (ssb_ddate.d_year = 1997 OR ssb_ddate.d_year = 1998)
--   AND ssb_part.p_category = 'MFGR#14'
-- GROUP BY ssb_ddate.d_year, ssb_supplier.s_city, ssb_part.p_brand1
-- ORDER BY ssb_ddate.d_year, ssb_supplier.s_city, ssb_part.p_brand1;

CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_orderdate, lo_custkey, lo_suppkey, lo_partkey);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_year);
CREATE INDEX temp_ix_3 ON ssb_customer(c_custkey, c_region);
CREATE INDEX temp_ix_4 ON ssb_supplier(s_suppkey, s_nation);
CREATE INDEX temp_ix_5 ON ssb_part(p_partkey, p_category);

UPDATE STATISTICS FOR ssb_lineorder(lo_orderdate, lo_custkey, lo_suppkey, lo_partkey);
UPDATE STATISTICS FOR ssb_ddate(d_datekey, d_year);
UPDATE STATISTICS FOR ssb_customer(c_custkey, c_region);
UPDATE STATISTICS FOR ssb_supplier(s_suppkey, s_nation);
UPDATE STATISTICS FOR ssb_part(p_partkey, p_category);

SELECT ssb_ddate.d_year, ssb_supplier.s_city, ssb_part.p_brand1, sum(ssb_lineorder.lo_revenue-ssb_lineorder.lo_supplycost) AS profit
FROM ssb_ddate
INNER JOIN ssb_lineorder ON  ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
INNER JOIN ssb_customer ON ssb_lineorder.lo_custkey = ssb_customer.c_custkey
INNER JOIN ssb_supplier ON ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey
INNER JOIN ssb_part ON ssb_lineorder.lo_partkey = ssb_part.p_partkey
WHERE 
  ssb_customer.c_region = 'AMERICA'
  AND ssb_supplier.s_nation = 'UNITED STATES'
  AND (ssb_ddate.d_year = 1997 OR ssb_ddate.d_year = 1998)
  AND ssb_part.p_category = 'MFGR#14'
GROUP BY ssb_ddate.d_year, ssb_supplier.s_city, ssb_part.p_brand1
ORDER BY ssb_ddate.d_year, ssb_supplier.s_city, ssb_part.p_brand1;

DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;
DROP INDEX ssb_customer.temp_ix_3;
DROP INDEX ssb_supplier.temp_ix_4;
DROP INDEX ssb_part.temp_ix_5;
