
/*+
HashJoin(ddate, lineorder)
Leading(lineorder, part, supplier, ddate)
SeqScan(ddate)
*/

SELECT SUM(lineorder.lo_revenue), ddate.d_year, part.p_brand1
FROM  lineorder, ddate, part, supplier
WHERE lineorder.lo_orderdate = ddate.d_datekey
  AND lineorder.lo_partkey = part.p_partkey
  AND lineorder.lo_suppkey = supplier.s_suppkey
  AND part.p_brand1 = 'MFGR#2221'
  AND supplier.s_region = 'EUROPE'
GROUP BY ddate.d_year, part.p_brand1
ORDER BY ddate.d_year, part.p_brand1;