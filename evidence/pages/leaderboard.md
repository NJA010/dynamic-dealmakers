---
title: Leaderboard Development
---

<!-- aggregated -->
```sql leaderboard
select
	scraped_at 
	, team_name 
	, percentile_cont(.5) within group(order by score) as score

from memory."dynamic_dealmakers".leaderboards l
where 
    (team_name in (${inputs.Competitor}) or ('') in (${inputs.Competitor}))
    and scraped_at > '2024-04-12 17:00:00'
group by scraped_at, team_name
order by scraped_at desc, team_name
```


<!-- filters -->
```sql competitors
select distinct
    team_name
from memory."dynamic_dealmakers".leaderboards l
order by team_name
```


## Filters 
<Multiselect
    data={competitors}
    name=Competitor
    value=team_name
    label=team_name
    title="Select a Team"
    />

## Prices by product
<LineChart 
    data={leaderboard}
    x=scraped_at
    y=score 
    series=team_name
    type=grouped
    yAxisTitle="Revenue (EUR)"
    xAxisTitle="Date"
    xFmt="mmmm d, yyyy H:MM:SS AM/PM"
    xAxisLabels=true
/>

<DataTable data={leaderboard} search=true sort=false>
    <Column id="scraped_at" title="scraped_at" fmt="mmmm d, yyyy H:MM:SS AM/PM" />
    <Column id="team_name" title="team_name" />
    <Column id="score" title="score" fmt='#,##0' />
</DataTable>

