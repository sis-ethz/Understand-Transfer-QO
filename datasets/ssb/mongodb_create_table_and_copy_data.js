use ssb

db.part.drop()
db.createCollection("part")
// CREATE TABLE part
//     (
//         p_partkey     INTEGER NOT NULL PRIMARY KEY,
//         p_name        VARCHAR(22) NOT NULL,
//         p_mfgr        VARCHAR(6) NOT NULL,
//         p_category    VARCHAR(7) NOT NULL,
//         p_brand1      VARCHAR(9) NOT NULL,
//         p_color       VARCHAR(11) NOT NULL,
//         p_type        VARCHAR(25) NOT NULL,
//         p_size        INTEGER NOT NULL,
//         p_container   VARCHAR(10) NOT NULL
//     );

db.supplier.drop()
db.createCollection("supplier")
// DROP TABLE IF EXISTS supplier;
// CREATE TABLE supplier
//     (
//         s_suppkey   INTEGER NOT NULL PRIMARY KEY,
//         s_name      VARCHAR(25) NOT NULL,
//         s_address   VARCHAR(25) NOT NULL,
//         s_city      VARCHAR(10) NOT NULL,
//         s_nation    VARCHAR(15) NOT NULL,
//         s_region    VARCHAR(12) NOT NULL,
//         s_phone     VARCHAR(15) NOT NULL
//     );

db.customer.drop()
db.createCollection("customer")
// DROP TABLE IF EXISTS customer;
// CREATE TABLE customer
//     (
//         c_custkey      INTEGER NOT NULL PRIMARY KEY,
//         c_name         VARCHAR(25) NOT NULL,
//         c_address      VARCHAR(25) NOT NULL,
//         c_city         VARCHAR(10) NOT NULL,
//         c_nation       VARCHAR(15) NOT NULL,
//         c_region       VARCHAR(12) NOT NULL,
//         c_phone        VARCHAR(15) NOT NULL,
//         c_mktsegment   VARCHAR(10) NOT NULL
//     );

db.ddate.drop()
db.createCollection("ddate")
// DROP TABLE IF EXISTS ddate;
// CREATE TABLE ddate
//     (
//         d_datekey            INTEGER NOT NULL,
//         d_date               VARCHAR(19) NOT NULL,
//         d_dayofweek          VARCHAR(10) NOT NULL,
//         d_month              VARCHAR(10) NOT NULL,
//         d_year               INTEGER NOT NULL,
//         d_yearmonthnum       INTEGER NOT NULL,
//         d_yearmonth          VARCHAR(8) NOT NULL,
//         d_daynuminweek       INTEGER NOT NULL,
//         d_daynuminmonth      INTEGER NOT NULL,
//         d_daynuminyear       INTEGER NOT NULL,
//         d_monthnuminyear     INTEGER NOT NULL,
//         d_weeknuminyear      INTEGER NOT NULL,
//         d_sellingseason      VARCHAR(13) NOT NULL,
//         d_lastdayinweekfl    VARCHAR(1) NOT NULL,
//         d_lastdayinmonthfl   VARCHAR(1) NOT NULL,
//         d_holidayfl          VARCHAR(1) NOT NULL,
//         d_weekdayfl          VARCHAR(1) NOT NULL
//     );

db.lineorder.drop()
db.createCollection("lineorder")
// DROP TABLE IF EXISTS lineorder;
// CREATE TABLE lineorder
//     (
//         lo_orderkey          INTEGER NOT NULL,
//         lo_linenumber        INTEGER NOT NULL,
//         lo_custkey           INTEGER NOT NULL,
//         lo_partkey           INTEGER NOT NULL,
//         lo_suppkey           INTEGER NOT NULL,
//         lo_orderdate         INTEGER NOT NULL,
//         lo_orderpriority     VARCHAR(15) NOT NULL,
//         lo_shippriority      VARCHAR(1) NOT NULL,
//         lo_quantity          INTEGER NOT NULL,
//         lo_extendedprice     INTEGER NOT NULL,
//         lo_ordertotalprice   INTEGER NOT NULL,
//         lo_discount          INTEGER NOT NULL,
//         lo_revenue           INTEGER NOT NULL,
//         lo_supplycost        INTEGER NOT NULL,
//         lo_tax               INTEGER NOT NULL,
//         lo_commitdate        INTEGER NOT NULL,
//         lo_shipmode          VARCHAR(10) NOT NULL
//     );

// GO


// USE ssb;



// BULK INSERT part
// FROM '/mnt/interpretable-cost-model/datasets/ssb/tables/part_fixed.tbl'
// WITH
//     (
//         FORMAT = 'CSV',
//         FIELDQUOTE = '"',
//         FIRSTROW = 1,
//         FIELDTERMINATOR = '|', --CSV field delimiter
//     ROWTERMINATOR = '\n', --Use to shift the control to next row
//     TABLOCK
//     );


// BULK INSERT supplier
// FROM '/mnt/interpretable-cost-model/datasets/ssb/tables/supplier_fixed.tbl'
// WITH
//     (
//         FORMAT = 'CSV',
//         FIELDQUOTE = '"',
//         FIRSTROW = 1,
//         FIELDTERMINATOR = '|', --CSV field delimiter
//     ROWTERMINATOR = '\n', --Use to shift the control to next row
//     TABLOCK
//     );

// BULK INSERT ddate
// FROM '/mnt/interpretable-cost-model/datasets/ssb/tables/date_fixed.tbl'
// WITH
//     (
//         FORMAT = 'CSV',
//         FIELDQUOTE = '"',
//         FIRSTROW = 1,
//         FIELDTERMINATOR = '|', --CSV field delimiter
//     ROWTERMINATOR = '\n', --Use to shift the control to next row
//     TABLOCK
//     );

// BULK INSERT customer
// FROM '/mnt/interpretable-cost-model/datasets/ssb/tables/customer_fixed.tbl'
// WITH
//     (
//         FORMAT = 'CSV',
//         FIELDQUOTE = '"',
//         FIRSTROW = 1,
//         FIELDTERMINATOR = '|', --CSV field delimiter
//     ROWTERMINATOR = '\n', --Use to shift the control to next row
//     TABLOCK
//     );

// BULK INSERT lineorder
// FROM '/mnt/interpretable-cost-model/datasets/ssb/tables/lineorder_fixed.tbl'
// WITH
//     (
//         FORMAT = 'CSV',
//         FIELDQUOTE = '"',
//         FIRSTROW = 1,
//         FIELDTERMINATOR = '|', --CSV field delimiter
//     ROWTERMINATOR = '\n', --Use to shift the control to next row
//     TABLOCK
//     );

// GO



