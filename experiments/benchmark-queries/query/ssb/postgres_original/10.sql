SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
FROM customer, lineorder, supplier, ddate   
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_orderdate = d_datekey
  AND (c_city = 'UNITED KI1' OR c_city = 'UNITED KI5')
  AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')
  AND d_yearmonth = 'Dec1997'
GROUP BY c_city, s_city, d_year
ORDER BY d_year ASC, revenue DESC;

   
-- SELECT COUNT(*) FROM customer WHERE 
-- (c_city = 'UNITED KI1' OR c_city = 'UNITED KI5')
-- AND c_custkey in (
-- SELECT lo_custkey
-- FROM lineorder, supplier, ddate   
-- WHERE lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey 
--   AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')
--   AND d_yearmonth = 'Dec1997')  ;


-- SELECT count(*) from ddate where d_datekey in 
-- (SELECT lo_orderdate 
-- FROM lineorder, supplier   
-- WHERE lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')) AND d_yearmonth = 'Dec1997';
  
-- SELECT count(*) from ddate where d_datekey in 
-- (SELECT lo_orderdate 
-- FROM lineorder, supplier   
-- WHERE lo_suppkey = s_suppkey
--   AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5'));
  

-- SELECT COUNT(*) FROM customer WHERE 
-- c_custkey in (
-- SELECT lo_custkey
-- FROM lineorder, supplier, ddate   
-- WHERE lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey 
--   AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')
--   AND d_yearmonth = 'Dec1997')  ;

