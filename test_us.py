import pandas as pd
pop = pd.read_csv('data-USstates-master/state-population.csv')
areas = pd.read_csv('data-USstates-master/state-areas.csv')
abbrevs = pd.read_csv('data-USstates-master/state-abbrevs.csv')

merged = pd.merge(pop, abbrevs, how='outer',
                  left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1)

merged.loc[merged['state/region']=='PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'


final = pd.merge(merged, areas, on='state', how='left')
# print(final)
final.dropna(inplace=True)

data2010 = final.query("year == 2010 & ages == 'total'")
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(inplace=True,ascending=False)

print(merged.loc[merged['state'].isnull()])