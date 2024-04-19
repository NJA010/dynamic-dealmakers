select distinct on (batch_key)
		batch_key
		, batch_id
		, product_name
from "dynamic-dealmakers".public.products
