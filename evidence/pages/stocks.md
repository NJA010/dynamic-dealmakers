


---
title: Stocks Development
---

<!-- aggregated -->

```sql stocks
with distinct_products as (
	select distinct on (batch_id)
		batch_id::integer as batch_id
		, product_name
	
	from memory."dynamic_dealmakers".products p
)
select
	s.scraped_at 
	, p.product_name 
	, sum(s.stock_amount) as stock_amount
from memory."dynamic_dealmakers".stocks s
left join distinct_products p using(batch_id)
where
    (p.product_name in (${inputs.Products}) or '' in (${inputs.Products}))
group by s.scraped_at, p.product_name
order by s.scraped_at desc, p.product_name
```


<!-- filters -->
```sql products
select distinct
    product_name
from memory."dynamic_dealmakers".products
order by product_name
```


## Filters 

<Multiselect
    data={products}
    name=Products
    value=product_name
    label=product_name
    title="Selecteer een product_name"
    />


## Prices by product

<LineChart 
    data={stocks}
    x=scraped_at
    y=stock_amount 
    series=product_name
    type=grouped
/>

<DataTable data={stocks} search=true sort=false>
    <Column id="scraped_at" title="scraped_at" fmt="mmmm d, yyyy H:MM:SS AM/PM" />
    <Column id="product_name" title="product_name" />
    <Column id="stock_amount" title="stock_amount" />
</DataTable>

