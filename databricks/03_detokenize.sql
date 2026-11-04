-- Detokenization in Unity Catalog (databricks/README.md, section 2).
-- Members of the account group spend_pii_readers see real identifiers; everyone else sees the token.
-- The token is base64(iv12 || ciphertext || tag16), the layout aes_decrypt(..., 'GCM') reads.
--
-- To verify on the workspace before relying on it: that secret() resolves inside a SQL UDF /
-- column mask for the calling user (it needs READ on the scope, granted to spend_pii_readers in
-- 01_setup.sql). If it does not, use the fallback at the end of this file.

-- 1. The function.
CREATE OR REPLACE FUNCTION <catalog>.spend_gov.detok(tok STRING)
  RETURNS STRING
  COMMENT 'Cloudera spend token -> identifier, for spend_pii_readers only'
  RETURN CASE
    WHEN tok IS NULL THEN NULL
    WHEN is_account_group_member('spend_pii_readers')
      THEN CAST(aes_decrypt(unbase64(tok), unbase64(secret('spend_tokens', 'enc_key_v1')), 'GCM') AS STRING)
    ELSE tok
  END;

-- 2. Views of the current rows (SCD2: __END_AT IS NULL), identifiers passed through detok().
CREATE OR REPLACE VIEW <catalog>.spend_silver_v.customer AS
SELECT <catalog>.spend_gov.detok(cif) AS cif, * EXCEPT (cif, __START_AT, __END_AT)
FROM <catalog>.spend_silver.customer WHERE __END_AT IS NULL;

CREATE OR REPLACE VIEW <catalog>.spend_silver_v.account AS
SELECT <catalog>.spend_gov.detok(account_id) AS account_id, <catalog>.spend_gov.detok(cif) AS cif,
       * EXCEPT (account_id, cif, __START_AT, __END_AT)
FROM <catalog>.spend_silver.account WHERE __END_AT IS NULL;

CREATE OR REPLACE VIEW <catalog>.spend_silver_v.spend_event AS
SELECT <catalog>.spend_gov.detok(event_id) AS event_id, <catalog>.spend_gov.detok(cif) AS cif,
       <catalog>.spend_gov.detok(account_id) AS account_id,
       * EXCEPT (event_id, cif, account_id, __START_AT, __END_AT)
FROM <catalog>.spend_silver.spend_event WHERE __END_AT IS NULL;

-- Join on tokens, decrypt at the end: detok() per row is the expensive part.
--   SELECT spend_gov.detok(c.cif) AS cif, sum(e.amount)
--   FROM spend_silver.spend_event e JOIN spend_silver.customer c USING (cif)
--   WHERE e.__END_AT IS NULL AND c.__END_AT IS NULL GROUP BY c.cif;

-- 3. Alternative: a column mask on the tables themselves (same function, the column is its first argument).
--    Pipelines own their streaming tables, so set masks in the pipeline's table definition or with
--    ALTER on a table you own; ABAC policies on a 'pii' tag do the same at catalog scale.
--   ALTER TABLE <catalog>.spend_silver.customer ALTER COLUMN cif SET MASK <catalog>.spend_gov.detok;

-- 4. Fallback if secret() is not allowed in the function: keep the key in a table only the
--    function owner can read, and let the function read it (the function runs with its owner's rights
--    on referenced tables).
--   CREATE TABLE <catalog>.spend_gov.token_keys (key_version STRING, enc_key_b64 STRING);
--   REVOKE ALL PRIVILEGES ON TABLE <catalog>.spend_gov.token_keys FROM `account users`;
--   ... RETURN CASE WHEN is_account_group_member('spend_pii_readers') THEN
--         CAST(aes_decrypt(unbase64(tok),
--              unbase64((SELECT enc_key_b64 FROM <catalog>.spend_gov.token_keys WHERE key_version = 'v1')),
--              'GCM') AS STRING) ELSE tok END;
