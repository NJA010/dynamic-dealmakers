INSERT INTO stocks
SELECT 
  id,
  scraped_at,
  p.key::INTEGER AS stock_update_id,
  pp.key::INTEGER AS batch_id,
  pp.value::INTEGER AS stock_amount
FROM 
  raw_stocks,
  jsonb_each(payload) AS p(key,  value),
  jsonb_each(p.value) AS pp(key,  value)
WHERE id > {{ max_id }};

