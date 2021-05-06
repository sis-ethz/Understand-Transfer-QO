SELECT SUM(lo_extendedprice*lo_discount) AS revenue
FROM lineorder, ddate
WHERE lo_orderdate = d_datekey
  AND d_weeknuminyear = 6
  AND d_year = 1994
  AND lo_discount BETWEEN 5 AND 7
  AND lo_quantity BETWEEN 36 AND 40;



-- select count(*) from ddate where d_datekey in 
-- (SELECT lo_orderdate
-- FROM lineorder
-- WHERE lo_discount BETWEEN 5 AND 7
--   AND lo_quantity BETWEEN 36 AND 40) AND 
--   d_weeknuminyear = 6
--   AND d_year = 1994;

-- select count(*) from ddate where d_datekey in 
-- (SELECT lo_orderdate
-- FROM lineorder
-- WHERE lo_discount BETWEEN 5 AND 7
--   AND lo_quantity BETWEEN 36 AND 40);