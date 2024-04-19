
select
    *
    , left(scraped_at::varchar, 16) as match_nk
from "dynamic-dealmakers".public.stocks
