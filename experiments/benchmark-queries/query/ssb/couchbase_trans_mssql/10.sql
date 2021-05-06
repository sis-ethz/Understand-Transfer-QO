-- SELECT c_city, ssb_supplier.s_city, ssb_ddate.d_year, SUM(ssb_lineorder.lorevenue) AS revenue
-- FROM customer, lineorder, supplier, ddate   
-- WHERE ssb_lineorder.locustkey = c_custkey
--   AND ssb_lineorder.losuppkey = ssb_supplier.s_suppkey
--   AND ssb_lineorder.loorderdate = ssb_ddate.d_datekey
--   AND (c_city = 'UNITED KI1' OR c_city = 'UNITED KI5')
--   AND (ssb_supplier.s_city = 'UNITED KI1' OR ssb_supplier.s_city = 'UNITED KI5')
--   AND ssb_ddate.d_yearmonth = 'Dec1997'
-- GROUP BY c_city, ssb_supplier.s_city, ssb_ddate.d_year
-- ORDER BY ssb_ddate.d_year ASC, revenue DESC;

CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_custkey, lo_orderdate, lo_suppkey);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_yearmonth);
CREATE INDEX temp_ix_3 ON ssb_customer(c_custkey, c_city);
CREATE INDEX temp_ix_4 ON ssb_supplier(s_suppkey, s_city);


SELECT ssb_customer.c_city, ssb_supplier.s_city, ssb_ddate.d_year, SUM(ssb_lineorder.lo_revenue) AS revenue
FROM ssb_customer 
INNER JOIN ssb_lineorder USE NL ON ssb_lineorder.lo_custkey = ssb_customer.c_custkey
INNER JOIN ssb_supplier USE HASH(BUILD) ON ssb_lineorder.lo_suppkey = ssb_supplier.s_suppkey
INNER JOIN ssb_ddate USE HASH(BUILD) ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
WHERE 
 (ssb_customer.c_city = 'UNITED KI1' OR ssb_customer.c_city = 'UNITED KI5')
  AND (ssb_supplier.s_city = 'UNITED KI1' OR ssb_supplier.s_city = 'UNITED KI5')
  AND ssb_ddate.d_yearmonth = 'Dec1997'
GROUP BY ssb_customer.c_city, ssb_supplier.s_city, ssb_ddate.d_year
ORDER BY ssb_ddate.d_year ASC, revenue DESC;


DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;
DROP INDEX ssb_customer.temp_ix_3;
DROP INDEX ssb_supplier.temp_ix_4;