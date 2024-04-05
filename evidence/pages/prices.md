---
title: Product Prices Development
---

<!-- aggregated -->
```sql prices
select 
    product_name
    , competitor_price
    , scraped_at
from memory."dynamic-dealmakers".prices
order by scraped_at
```


<!-- filters -->
```sql products
select distinct
    product_name
from memory."dynamic-dealmakers".prices
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
    data={prices}
    series=product_name
    type=competitor_price
    sort=false
    yAxisTitle=price
    />

<DataTable data={prices} search=true>
    <Column id="product_name" title="product_name" />
    <Column id="competitor_price" title="competitor_price" />
    <Column id="scraped_at" title="scraped_at" />
</DataTable>

