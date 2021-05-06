-- SELECT SUM(lo_extendedprice*lo_discount) AS revenue
-- FROM lineorder, ddate
-- WHERE lo_orderdate = d_datekey
--   AND d_weeknuminyear = 6
--   AND d_year = 1994
--   AND lo_discount BETWEEN 5 AND 7
--   AND lo_quantity BETWEEN 36 AND 40;


CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_discount, lo_quantity,lo_orderdate);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_weeknuminyear, d_year);

UPDATE STATISTICS FOR ssb_lineorder (lo_discount, lo_quantity,lo_orderdate);
UPDATE STATISTICS FOR ssb_ddate (d_datekey, d_weeknuminyear, d_year);

SELECT SUM(ssb_lineorder.lo_extendedprice*ssb_lineorder.lo_discount) AS revenue
FROM ssb_lineorder 
INNER JOIN ssb_ddate ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey

WHERE ssb_ddate.d_weeknuminyear = 6
  AND ssb_ddate.d_year = 1994
  AND ssb_lineorder.lo_discount BETWEEN 5 AND 7
  AND ssb_lineorder.lo_quantity BETWEEN 36 AND 40;


DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;

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