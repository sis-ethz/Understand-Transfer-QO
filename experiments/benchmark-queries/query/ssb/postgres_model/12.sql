/*+
HashJoin(lineorder supplier)
HashJoin( lineorder supplier customer) 
HashJoin(lineorder supplier customer ddate)
MergeJoin(lineorder supplier customer ddate part)
IndexScan(part)
Leading(lineorder supplier customer ddate part)
*/

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
