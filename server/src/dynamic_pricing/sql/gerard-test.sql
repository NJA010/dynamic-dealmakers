--select count(*) from raw_prices rp; 
--select count(*) from raw_products rp;
--select count(*) from raw_leaderboards rl;
--select count(*) from raw_stocks rs;

select * 
--from raw_stocks 
from raw_prices 
--from raw_products 
--from raw_leaderboards 
order by scraped_at desc;

DELETE FROM raw_prices 
WHERE payload = '{"detail": "Not authenticated"}';


