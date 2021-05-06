SELECT SUM(lo_extendedprice*lo_discount) AS revenue
FROM lineorder, ddate
WHERE lo_orderdate = d_datekey
  AND d_yearmonthnum = 199401
  AND lo_discount BETWEEN 4 AND 6
  AND lo_quantity BETWEEN 26 AND 35;


-- select count(*) from ddate where d_datekey in 
-- (SELECT lo_orderdate
-- FROM lineorder
-- WHERE lo_discount BETWEEN 4 AND 6
--   AND lo_quantity BETWEEN 26 AND 35
-- ) AND d_yearmonthnum = 199401;

-- select count(*) from ddate where d_datekey in 
-- (SELECT lo_orderdate
-- FROM lineorder
-- WHERE lo_discount BETWEEN 4 AND 6
--   AND lo_quantity BETWEEN 26 AND 35
-- );