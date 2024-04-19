
select
    *
    , left(scraped_at::varchar, 16) as match_nk
    , batch_name as batch_key
from "dynamic-dealmakers".public.prices

