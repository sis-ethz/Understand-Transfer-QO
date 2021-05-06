SELECT SUM(lo_revenue), d_year, p_brand1
FROM  lineorder, ddate, part, supplier
WHERE lo_orderdate = d_datekey
  AND lo_partkey = p_partkey
  AND lo_suppkey = s_suppkey
  AND p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'
  AND s_region = 'AMERICA'
GROUP BY d_year, p_brand1
ORDER BY d_year, p_brand1;


-- SELECT COUNT(*) FROM ddate WHERE d_datekey in (
-- SELECT lo_orderdate
-- FROM  lineorder, part, supplier
-- WHERE lo_partkey = p_partkey
--   AND lo_suppkey = s_suppkey
--   AND p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'
--   AND s_region = 'AMERICA'
-- );

-- SELECT COUNT(*) FROM part WHERE p_partkey IN (
-- SELECT lo_partkey
-- FROM  lineorder) AND p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228';

-- SELECT COUNT(*) FROM ddate WHERE d_datekey in (
-- SELECT lo_orderdate
-- FROM  lineorder, part, supplier
-- WHERE lo_partkey = p_partkey
--   AND lo_suppkey = s_suppkey
--   AND p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'
--   AND s_region = 'AMERICA'
-- );
