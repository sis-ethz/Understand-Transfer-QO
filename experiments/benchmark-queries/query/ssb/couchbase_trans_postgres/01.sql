-- SELECT SUM(lo_extendedprice*lo_discount) AS revenue
-- FROM lineorder, ddate
-- WHERE lo_orderdate = d_datekey
--   AND d_year = 1993
--   AND lo_discount BETWEEN 1 AND 3
--   AND lo_quantity < 25;

-- Couchbase Oringinal + Our knowledge

CREATE INDEX temp_ix_1 ON ssb_lineorder(lo_discount, lo_quantity,lo_orderdate);
CREATE INDEX temp_ix_2 ON ssb_ddate(d_datekey, d_year);

SELECT SUM(ssb_lineorder.lo_extendedprice * ssb_lineorder.lo_discount) AS revenue
FROM ssb_lineorder INNER JOIN ssb_ddate USE HASH(BUILD) ON  ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
WHERE ssb_ddate.d_year = 1993
AND ssb_lineorder.lo_discount BETWEEN 1 AND 3
AND ssb_lineorder.lo_quantity < 25;

DROP INDEX ssb_lineorder.temp_ix_1;
DROP INDEX ssb_ddate.temp_ix_2;