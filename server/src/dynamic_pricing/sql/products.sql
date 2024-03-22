INSERT INTO products
WITH products AS (
  SELECT
    id,
    scraped_at,
    p.key AS product_name,
    jsonb_object_agg(pp.key, pp.value) AS batch_info
  FROM
    raw_products,
    jsonb_each(payload) AS p(key, value),
    jsonb_each(p.value->'products') AS pp(key, value)
  WHERE id > {{ max_id }}
  GROUP BY 1, 2, 3
)

SELECT
  id,
  scraped_at,
  product_name,
  p.key AS batch_name,
  p.value->>'id' AS batch_id,
  (p.value->>'sell_by')::TIMESTAMP AS batch_expiry_date
FROM
  products,
  jsonb_each(batch_info) AS p(key, value);
