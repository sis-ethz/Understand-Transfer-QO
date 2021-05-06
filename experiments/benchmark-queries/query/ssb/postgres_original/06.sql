SELECT SUM(lineorder.lo_revenue), ddate.d_year, part.p_brand1
FROM  lineorder, ddate, part, supplier
WHERE lineorder.lo_orderdate = ddate.d_datekey
  AND lineorder.lo_partkey = part.p_partkey
  AND lineorder.lo_suppkey = supplier.s_suppkey
  AND part.p_brand1 = 'MFGR#2221'
  AND supplier.s_region = 'EUROPE'
GROUP BY ddate.d_year, part.p_brand1
ORDER BY ddate.d_year, part.p_brand1;


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