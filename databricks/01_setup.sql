-- One-time Unity Catalog setup for the Cloudera spend export (databricks/README.md).
-- Run as a metastore / workspace admin. Replace <catalog> and <iam-role-arn>.

-- 1. Read access to the export in the Cloudera project bucket. The IAM role needs
--    s3:GetObject and s3:ListBucket on the prefix below (read-only: Databricks never writes there).
CREATE STORAGE CREDENTIAL IF NOT EXISTS spend_cloudera_export
  WITH (AWS_IAM_ROLE = '<iam-role-arn>')            -- or create it in Catalog Explorer
  COMMENT 'Read-only: Cloudera spend export';

CREATE EXTERNAL LOCATION IF NOT EXISTS spend_cloudera_export
  URL 's3://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export/'
  WITH (STORAGE CREDENTIAL spend_cloudera_export)
  COMMENT 'Tokenized Parquet change batches from cde/jobs/export_parquet.py';

-- 2. Catalog and schemas: raw batches, SCD2 tables (tokens), views (detokenized for members).
CREATE CATALOG IF NOT EXISTS <catalog>;
CREATE SCHEMA IF NOT EXISTS <catalog>.spend_bronze   COMMENT 'Raw change batches as exported';
CREATE SCHEMA IF NOT EXISTS <catalog>.spend_silver   COMMENT 'SCD type 2 tables, identifiers tokenized';
CREATE SCHEMA IF NOT EXISTS <catalog>.spend_silver_v COMMENT 'Current rows; identifiers decrypted for spend_pii_readers';
CREATE SCHEMA IF NOT EXISTS <catalog>.spend_gov      COMMENT 'detok() and masks';

-- 3. The decryption key: enc_key only (never iv_key). Databricks CLI, not SQL:
--
--   databricks secrets create-scope spend_tokens
--   databricks secrets put-secret spend_tokens enc_key_v1 --string-value "<enc_key_b64>"
--   databricks secrets put-acl spend_tokens spend_pii_readers READ
--
--   <enc_key_b64> is the enc_key_b64 field of the CDE workload credential
--   rsingh-spend-anl-token-key (backup: ~/.spend_tokens/rsingh-spend-anl-token-key-v1.json).
--   Pass it from a file or a prompt, not on a shared shell history.
--   After a rotation add enc_key_v2 next to v1; keep v1 until no table holds v1 tokens.

-- 4. Who may see real identifiers: an account-level group (is_account_group_member checks account groups).
--    Create spend_pii_readers in the account console and add members there.

-- 5. Grants: analysts read tokens, pipeline owner writes, only spend_pii_readers decrypt.
GRANT READ FILES ON EXTERNAL LOCATION spend_cloudera_export TO `spend_pipeline_owners`;
GRANT USE CATALOG ON CATALOG <catalog> TO `spend_analysts`;
GRANT USE SCHEMA, SELECT ON SCHEMA <catalog>.spend_silver   TO `spend_analysts`;
GRANT USE SCHEMA, SELECT ON SCHEMA <catalog>.spend_silver_v TO `spend_analysts`;
GRANT USE SCHEMA, EXECUTE ON SCHEMA <catalog>.spend_gov     TO `spend_analysts`;
