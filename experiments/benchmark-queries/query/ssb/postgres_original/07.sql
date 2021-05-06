SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
FROM customer, lineorder, supplier, ddate   
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_orderdate = d_datekey
  AND c_region = 'ASIA'
  AND s_region = 'ASIA'
  AND d_year >= 1992 AND d_year <= 1997
GROUP BY c_nation, s_nation, d_year
ORDER BY d_year ASC, revenue DESC;



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

