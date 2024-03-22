INSERT INTO prices
WITH grouped_prices AS (
  SELECT
    id,
    scraped_at,
    p.key AS product_name,
    jsonb_object_agg(pp.key, pp.value) AS batch_info
  FROM
    raw_prices,
    jsonb_each(payload) AS p(key, value),
    jsonb_each(p.value) AS pp(key, value)
  WHERE id > {{ max_id }}
  GROUP BY 1, 2, 3
),

prices AS (
  SELECT
    id,
    scraped_at,
    product_name,
    p.key AS batch_name,
    p.value AS competitor_prices
  FROM
    grouped_prices,
    jsonb_each(batch_info) AS p(key, value)
)

SELECT
  id,
  scraped_at,
  product_name,
  batch_name,
  p.key AS competitor_name,
  p.value::NUMERIC AS competitor_price
FROM
  prices,
  jsonb_each(competitor_prices) AS p(key, value)

