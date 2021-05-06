-- SELECT SUM(lineorder.lo_revenue), ddate.d_year, part.p_brand1
-- FROM  lineorder, ddate, part, supplier
-- WHERE lineorder.lo_orderdate = ddate.d_datekey
--   AND lineorder.lo_partkey = part.p_partkey
--   AND lineorder.lo_suppkey = supplier.s_suppkey
--   AND part.p_brand1 = 'MFGR#2221'
--   AND supplier.s_region = 'EUROPE'
-- GROUP BY ddate.d_year, part.p_brand1
-- ORDER BY ddate.d_year, part.p_brand1;

CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_orderdate, lo_partkey, lo_suppkey);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_year);
CREATE INDEX temp_ix_3 ON ssb_part(p_partkey, p_brand1);
CREATE INDEX temp_ix_4 ON ssb_supplier(s_suppkey, s_region);

SELECT SUM(ssb_lineorder.lo_revenue), ssb_ddate.d_year, ssb_part.p_brand1
FROM  ssb_lineorder 
INNER JOIN ssb_ddate USE HASH(BUILD) ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
INNER JOIN ssb_part USE HASH(BUILD) ON ssb_lineorder.lo_partkey = ssb_part.p_partkey
INNER JOIN ssb_supplier USE HASH(BUILD) ON ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey 
WHERE 
  ssb_part.p_brand1 = 'MFGR#2221'
  AND ssb_supplier.s_region = 'EUROPE'
GROUP BY ssb_ddate.d_year, ssb_part.p_brand1
ORDER BY ssb_ddate.d_year, ssb_part.p_brand1;

DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;
DROP INDEX ssb_part.temp_ix_3;
DROP INDEX ssb_supplier.temp_ix_4;

-- SELECT COUNT(*) FROM part WHERE p_partkey IN (
-- SELECT lo_partkey
-- FROM  lineorder)
-- AND part.p_brand1 = 'MFGR#2221';

-- SELECT COUNT(*) FROM ddate where d_datekey in
-- (SELECT lo_orderdate
-- FROM  lineorder, part, supplier
-- WHERE lineorder.lo_partkey = part.p_partkey
--   AND lineorder.lo_suppkey = supplier.s_suppkey
--   AND part.p_brand1 = 'MFGR#2221'
--   AND supplier.s_region = 'EUROPE')
--   ;

-- SELECT COUNT(*) FROM part WHERE p_partkey IN (
-- SELECT lo_partkey
-- FROM  lineorder);