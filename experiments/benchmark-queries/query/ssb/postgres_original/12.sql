SELECT d_year, s_nation, p_category, sum(lo_revenue-lo_supplycost) AS profit
FROM ddate, customer, supplier, part, lineorder
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_partkey = p_partkey
  AND lo_orderdate = d_datekey
  AND c_region = 'AMERICA'
  AND s_region = 'AMERICA'
  AND (d_year = 1997 OR d_year = 1998)
  AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
GROUP BY d_year, s_nation, p_category
ORDER BY d_year, s_nation, p_category;


-- SELECT count(*) from part where 
-- (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2') AND p_partkey in (
--   SELECT lo_partkey
-- FROM ddate, customer, supplier, part, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND c_region = 'AMERICA'
--   AND s_region = 'AMERICA'
--   AND (d_year = 1997 OR d_year = 1998)
-- );


-- select count(*) from ddate where d_datekey in
-- (SELECT lo_orderdate
-- FROM customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_region = 'AMERICA'
--   AND s_region = 'AMERICA')
-- AND (d_year = 1997 OR d_year = 1998);

-- select count(*) from customer where c_custkey in 
-- (SELECT lo_custkey
-- FROM 
-- supplier, lineorder
-- WHERE lo_suppkey = s_suppkey
--   AND s_region = 'AMERICA')
-- AND c_region = 'AMERICA';


-- =================================================

-- select count(*) from customer where c_custkey in 
-- (SELECT lo_custkey
-- FROM 
-- supplier, lineorder
-- WHERE lo_suppkey = s_suppkey
--   AND s_region = 'AMERICA');

-- select count(*) from ddate where d_datekey in
-- (SELECT lo_orderdate
-- FROM customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND c_region = 'AMERICA'
--   AND s_region = 'AMERICA');

-- SELECT count(*) from part where 
-- p_partkey in (
--   SELECT lo_partkey
-- FROM ddate, customer, supplier, lineorder
-- WHERE lo_custkey = c_custkey
--   AND lo_suppkey = s_suppkey
--   AND lo_orderdate = d_datekey
--   AND c_region = 'AMERICA'
--   AND s_region = 'AMERICA'
--   AND (d_year = 1997 OR d_year = 1998)
-- );





