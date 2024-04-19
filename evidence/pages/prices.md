---
title: Product Prices Development
---
<!-- add boxes min max  -->
<!-- aggregated -->
```sql prices
select 
    product_name
    , scraped_at
    , max(competitor_name) as competitor_name
    , min(competitor_price) as min_competitor_price
    , max(competitor_price) as max_competitor_price
    , avg(competitor_price) as avg_competitor_price
    , median(competitor_price) as median_competitor_price
from  memory."dynamic_dealmakers".prices pri
where true
    and (competitor_name in (${inputs.Competitor}) or '' in (${inputs.Competitor}))
    and product_name like '${inputs.Products}'
group by product_name, scraped_at
order by scraped_at desc, competitor_name, product_name
```

```sql sold_stock
select
	pri.scraped_at
	, bm.product_name
	, sum(case 
            when pri.competitor_name = 'DynamicDealmakers'
            then sto.sold_stock * pri.competitor_price end
        ) as revenue
	, sum(sto.sold_stock) as quantity
    , min(pri.competitor_price) as lowest_price
    , min(case 
            when pri.competitor_name = 'DynamicDealmakers'
            then pri.competitor_price end
        ) as own_lowest_price
    , max(pri.competitor_price) as highest_price
from memory."dynamic_dealmakers".prices pri
left join memory."dynamic_dealmakers".batch_mapper bm
using(batch_key)
left join memory."dynamic_dealmakers".stocks sto
using(batch_id, match_nk)
where true
    and bm.product_name like '${inputs.Products}'
group by pri.scraped_at, bm.product_name
order by pri.scraped_at desc, bm.product_name
```

<!-- filters -->
```sql products
select distinct
    product_name
    , upper(product_name) as product_name_label
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
    data={competitors}
    name=Competitor
    value=competitor_name
    label=competitor_name
    title="Selecteer een competitor"
/>

<Dropdown
    data={products} 
    name=Products
    value=product_name
    label=product_name_label
    title="Selecteer een product_name"
    >
<DropdownOption valueLabel="All" value="%" />
</Dropdown>

<Dropdown name=Aggregation title="Select an aggregation type">
    <DropdownOption valueLabel=Median value="median_competitor_price" />
    <DropdownOption valueLabel=Min value="min_competitor_price" />
    <DropdownOption valueLabel=Max value="max_competitor_price" />
    <DropdownOption valueLabel=Average value="avg_competitor_price" />
</Dropdown>

## Prices by product
<LineChart 
    data={prices}
    x=scraped_at
    y={inputs.Aggregation}
    series=product_name
    type=grouped
/>

<DataTable data={prices} search=true sort=false>
    <Column id=product_name title=product_name />
    <Column id=scraped_at title=scraped_at />
    <Column id=competitor_name title=competitor_name />
    <Column id=min_competitor_price title=min_competitor_price />
    <Column id=max_competitor_price title=max_competitor_price />
    <Column id=avg_competitor_price title=avg_competitor_price />
    <Column id=median_competitor_price title=median_competitor_price />
</DataTable>


## Best prices to sales ratio (DynamicDealmakers)
{#if inputs.Products !== "%"}
### Product: {inputs.Products}

<LineChart 
    data={sold_stock} 
    x=scraped_at
    y={['own_lowest_price', 'lowest_price']}
    y2=revenue
    y2SeriesType=bar
    xAxisTitle="Time"
/>

{:else }

> ***NOTE***
Please set a filter on `products` in order to see data here.

{/if}