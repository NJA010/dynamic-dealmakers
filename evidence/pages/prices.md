---
title: Product Prices Development
---

<!-- aggregated -->
```sql prices
select 
    product_name
    , scraped_at
    , max(competitor_name) as competitor_name
    , median(competitor_price) as competitor_price
from  memory."dynamic_dealmakers".prices
where
    (product_name in (${inputs.Products}) or '' in (${inputs.Products}))
    and competitor_name like '${inputs.Competitor}'
group by product_name, scraped_at
order by scraped_at, competitor_name
```


<!-- filters -->
```sql products
select distinct
    product_name
from  memory."dynamic_dealmakers".prices
order by product_name
```

```sql competitors
select distinct
    competitor_name
from  memory."dynamic_dealmakers".prices
order by competitor_name
```


## Filters 

<Multiselect
    data={products}
    name=Products
    value=product_name
    label=product_name
    title="Selecteer een product_name"
    />

<Dropdown
    data={competitors}
    name=Competitor
    value=competitor_name
    title="Select a competitor"
    >
    <DropdownOption valueLabel="All" value="%" />
</Dropdown>


## Prices by product

<LineChart 
    data={prices}
    x=scraped_at
    y=competitor_price 
    series=product_name
    type=grouped
/>

<DataTable data={prices} search=true>
    <Column id="product_name" title="product_name" />
    <Column id="competitor_price" title="competitor_price" />
    <Column id="competitor_name" title="competitor_name" />
    <Column id="scraped_at" title="scraped_at" />
</DataTable>

