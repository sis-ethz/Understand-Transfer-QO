SELECT d_year, s_city, p_brand1, sum(lo_revenue-lo_supplycost) AS profit
FROM ddate, customer, supplier, part, lineorder
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_partkey = p_partkey
  AND lo_orderdate = d_datekey
  AND c_region = 'AMERICA'
  AND s_nation = 'UNITED STATES'
  AND (d_year = 1997 OR d_year = 1998)
  AND p_category = 'MFGR#14'
GROUP BY d_year, s_city, p_brand1
ORDER BY d_year, s_city, p_brand1;

-- SELECT COUNT(*) FROM part WHERE partkey IN
-- (SELECT lo_partkey 
-- FROM ddate, customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND c_region = 'AMERICA'
--   AND s_nation = 'UNITED STATES'
--   AND (d_year = 1997 OR d_year = 1998))
--   AND p_category = 'MFGR#14';


-- select count(*) from ddate where d_datekey in
-- (
-- SELECT lo_orderdate 
-- FROM customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_region = 'AMERICA'
--   AND s_nation = 'UNITED STATES'
-- )
--   AND (d_year = 1997 OR d_year = 1998);


-- select count(*) from customer where c_custkey in
-- (SELECT lo_custkey
-- FROM supplier, lineorder
-- WHERE lo_suppkey = s_suppkey
--   AND s_nation = 'UNITED STATES')
--   AND c_region = 'AMERICA';


----------------------------------------------------------------

-- select count(*) from customer where c_custkey in
-- (SELECT lo_custkey
-- FROM supplier, lineorder
-- WHERE lo_suppkey = s_suppkey
--   AND s_nation = 'UNITED STATES');


-- select count(*) from ddate where d_datekey in
-- (
-- SELECT lo_orderdate 
-- FROM customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_region = 'AMERICA'
--   AND s_nation = 'UNITED STATES'
-- );


-- SELECT COUNT(*) FROM part WHERE p_partkey IN
-- (SELECT lo_partkey 
-- FROM ddate, customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND c_region = 'AMERICA'
--   AND s_nation = 'UNITED STATES'
--   AND (d_year = 1997 OR d_year = 1998));


