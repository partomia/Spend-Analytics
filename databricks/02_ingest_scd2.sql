-- Lakeflow (Declarative) pipeline: Cloudera change batches -> SCD type 2 Delta tables.
-- Pipeline settings: catalog <catalog>, default schema spend_silver, triggered (run after the
-- nightly export lands, ~22:00 UTC), and the configuration values
--   spend.root         s3://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export
--   spend.start_batch  1        -- after a resync batch N: set N and run a FULL REFRESH (README 5)
--
-- AUTO CDC INTO is the current name of APPLY CHANGES INTO (the older syntax is still accepted).
-- SEQUENCE BY _batch: each key appears at most once per batch, so __START_AT / __END_AT are batch numbers.

-- ------------------------------------------------------------ customer (KEYS cif)
CREATE OR REFRESH STREAMING TABLE spend_bronze.customer_changes
  COMMENT 'Raw tokenized change batches of silver.customer'
AS SELECT *, _metadata.file_path AS _file
FROM STREAM read_files('${spend.root}/silver/customer/data/', format => 'parquet')
WHERE _batch >= ${spend.start_batch};

CREATE OR REFRESH STREAMING TABLE spend_silver.customer
  COMMENT 'silver.customer, SCD type 2; cif tokenized';

CREATE FLOW customer_scd2 AS AUTO CDC INTO spend_silver.customer
FROM STREAM(spend_bronze.customer_changes)
KEYS (cif)
APPLY AS DELETE WHEN _op = 'D'
SEQUENCE BY _batch
COLUMNS * EXCEPT (_op, _source_snapshot_id, _exported_at, _file)
STORED AS SCD TYPE 2;

-- ------------------------------------------------------------ account (KEYS account_id)
CREATE OR REFRESH STREAMING TABLE spend_bronze.account_changes
  COMMENT 'Raw tokenized change batches of silver.account'
AS SELECT *, _metadata.file_path AS _file
FROM STREAM read_files('${spend.root}/silver/account/data/', format => 'parquet')
WHERE _batch >= ${spend.start_batch};

CREATE OR REFRESH STREAMING TABLE spend_silver.account
  COMMENT 'silver.account, SCD type 2; account_id and cif tokenized';

CREATE FLOW account_scd2 AS AUTO CDC INTO spend_silver.account
FROM STREAM(spend_bronze.account_changes)
KEYS (account_id)
APPLY AS DELETE WHEN _op = 'D'
SEQUENCE BY _batch
COLUMNS * EXCEPT (_op, _source_snapshot_id, _exported_at, _file)
STORED AS SCD TYPE 2;

-- ------------------------------------------------------------ spend_event (KEYS event_id)
-- The largest table (about 30 M rows); clustered by date for the usual date-range reads.
CREATE OR REFRESH STREAMING TABLE spend_bronze.spend_event_changes
  COMMENT 'Raw tokenized change batches of silver.spend_event'
AS SELECT *, _metadata.file_path AS _file
FROM STREAM read_files('${spend.root}/silver/spend_event/data/', format => 'parquet')
WHERE _batch >= ${spend.start_batch};

CREATE OR REFRESH STREAMING TABLE spend_silver.spend_event
  COMMENT 'silver.spend_event, SCD type 2; event_id, cif and account_id tokenized'
  CLUSTER BY (txn_date);

CREATE FLOW spend_event_scd2 AS AUTO CDC INTO spend_silver.spend_event
FROM STREAM(spend_bronze.spend_event_changes)
KEYS (event_id)
APPLY AS DELETE WHEN _op = 'D'
SEQUENCE BY _batch
COLUMNS * EXCEPT (_op, _source_snapshot_id, _exported_at, _file)
STORED AS SCD TYPE 2;

-- Check after each run (README 7): current rows per table vs the manifests' counts.
--   SELECT count(*) FROM spend_silver.customer WHERE __END_AT IS NULL;
--   SELECT batch, batch_type, counts FROM read_files('${spend.root}/silver/customer/_manifests/', format => 'json');
