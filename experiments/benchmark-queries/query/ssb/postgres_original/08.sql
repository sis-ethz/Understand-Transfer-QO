SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
FROM customer, lineorder, supplier, ddate   
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_orderdate = d_datekey
  AND c_nation = 'UNITED STATES'
  AND s_nation = 'UNITED STATES'
  AND d_year >= 1992 AND d_year <= 1997
GROUP BY c_city, s_city, d_year
ORDER BY d_year ASC, revenue DESC;


-- SELECT COUNT(*) FROM customer WHERE c_custkey in (
--     SELECT lo_custkey
--   FROM lineorder, supplier   
--   WHERE lo_custkey = c_custkey
-- ) AND c_nation = 'UNITED STATES';

-- select count(*) from ddate where d_datekey in
-- (SELECT lo_orderdate
-- FROM customer, lineorder, supplier 
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_nation = 'UNITED STATES'
--   AND s_nation = 'UNITED STATES')
--   AND d_year >= 1992 AND d_year <= 1997;

-- SELECT COUNT(*) FROM customer WHERE c_custkey in (
--     SELECT lo_custkey
--   FROM lineorder, supplier   
--   WHERE lo_suppkey = s_suppkey
-- );

-- select count(*) from ddate where d_datekey in
-- (SELECT lo_orderdate
-- FROM customer, lineorder, supplier 
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_nation = 'UNITED STATES'
--   AND s_nation = 'UNITED STATES');