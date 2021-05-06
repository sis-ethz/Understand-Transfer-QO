-- SELECT d_year, c_nation, sum(lo_revenue-lo_supplycost) AS profit
-- FROM ddate, customer, supplier, part, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_partkey = p_partkey
--   AND lo_orderdate = d_datekey
--   AND c_region = 'AMERICA'
--   AND s_region = 'AMERICA'
--   AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
-- GROUP BY d_year, c_nation
-- ORDER BY d_year, c_nation;

CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_orderdate, lo_custkey, lo_suppkey, lo_partkey);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey);
CREATE INDEX temp_ix_3 ON ssb_customer(c_custkey, c_region);
CREATE INDEX temp_ix_4 ON ssb_supplier(s_suppkey, s_region);
CREATE INDEX temp_ix_5 ON ssb_part(p_partkey, p_mfgr);

UPDATE STATISTICS FOR ssb_lineorder(lo_orderdate, lo_custkey, lo_suppkey, lo_partkey);
UPDATE STATISTICS FOR ssb_ddate(d_datekey);
UPDATE STATISTICS FOR ssb_customer(c_custkey, c_region);
UPDATE STATISTICS FOR ssb_supplier(s_suppkey, s_region);
UPDATE STATISTICS FOR ssb_part(p_partkey, p_mfgr);


SELECT ssb_ddate.d_year, ssb_customer.c_nation, sum(ssb_lineorder.lo_revenue-ssb_lineorder.lo_supplycost) AS profit
FROM ssb_ddate
INNER JOIN ssb_lineorder ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
INNER JOIN ssb_customer ON ssb_lineorder.lo_custkey = ssb_customer.c_custkey
INNER JOIN ssb_supplier ON ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey
INNER JOIN ssb_part ON ssb_lineorder.lo_partkey = ssb_part.p_partkey
WHERE ssb_customer.c_region = 'AMERICA'
  AND ssb_supplier.s_region = 'AMERICA'
  AND (ssb_part.p_mfgr = 'MFGR#1' OR ssb_part.p_mfgr = 'MFGR#2')
GROUP BY ssb_ddate.d_year, ssb_customer.c_nation
ORDER BY ssb_ddate.d_year, ssb_customer.c_nation;

DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;
DROP INDEX ssb_customer.temp_ix_3;
DROP INDEX ssb_supplier.temp_ix_4;
DROP INDEX ssb_part.temp_ix_5;
