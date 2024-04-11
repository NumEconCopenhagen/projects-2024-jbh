import polars as pl
import geopandas as gpd
import pandas as pd
##############################################################################################################
#### Below is code for converting the "raw" ARCOS data to .parquet to save on space and speed up code ########
##############################################################################################################

# lf_all = (pl.scan_csv('data/raw/arcos_all_washpost.tsv', separator='\t', dtypes={'TRANSACTION_DATE':pl.String, 'REPORTER_BUS_ACT':pl.Categorical, 'REPORTER_ZIP':pl.String, 'BUYER_BUS_ACT':pl.Categorical, 'BUYER_STATE':pl.Categorical, 'TRANSACTION_CODE':pl.Categorical, 'BUYER_ZIP':pl.String, 'DRUG_CODE':pl.Categorical, 'NDC_NO':pl.String, 'DRUG_NAME':pl.Categorical, 'TRANSACTION_ID':pl.String, 'Measure':pl.Categorical, 'Product_Name': pl.Categorical, 'Ingredient_Name': pl.Categorical}, ignore_errors=True)
#       .select(pl.all().name.to_lowercase())
#       .with_columns(pl.col("transaction_date").str.pad_start(8, '0'))
#       .with_columns(pl.col("transaction_date").str.to_date(format='%m%d%Y'))
#       .select(pl.all().shrink_dtype())
# )
# lf_all.sink_parquet('data/raw/arcos_all_washpost.pq')

lf_all = pl.scan_parquet('data/raw/arcos_all_washpost.pq')

# Filtering for the Central Appalachian states:
## VA: Virginia, WV: West Virginia, KY: Kentucky, TN: Tennessee, NC: North Carolina.
# Geodata
states = ['VA', 'WV', 'KY', 'TN', 'NC']
states_fips = ['51', '54', '21', '47', '37']

# Make a list of the FIPS-codes the Appalachians (https://github.com/mkiang/narcan/blob/master/R/data-appalachia_fips.R)
appa_count_list = pl.read_csv('data/appalachia_fips.csv')['fipschar'].cast(pl.String).to_list()

# Read in shapefile of US counties and look at the counties in the "core"/central area of the Appalachians (https://www.arc.gov/about-the-appalachian-region/)
gdf_counties = gpd.read_file('data/raw/tl_2019_us_county/tl_2019_us_county.shp')
gdf_counties.columns = [name.lower() for name in gdf_counties.columns]
gdf_counties = gdf_counties[['statefp', 'countyfp', 'geoid', 'name', 'namelsad', 'geometry']]

# Save all US counties 
gdf_counties.to_parquet('data/us_counties.pq')

# Save Appalachians
gdf_counties_appa = gdf_counties[gdf_counties['statefp'].isin(states_fips)]
gdf_counties_appa = gdf_counties_appa[gdf_counties_appa['geoid'].isin(appa_count_list)]
gdf_counties_appa.reset_index(drop=True).to_parquet('data/us_appa_counties.pq')

# Do the same, but look at states
gdf_states = gpd.read_file('data/raw/cb_2018_us_state_500k.shp')
gdf_states.columns = [name.lower() for name in gdf_states.columns]

# Save all US states 
gdf_states.to_parquet('data/us_states.pq')

# Save Appalachians
gdf_states = gdf_states[gdf_states['statefp'].isin(states_fips)]
gdf_states.reset_index(drop=True).to_parquet('data/us_appa_states.pq')

# Filter for those pharmacies in the Central Appalachians
df_pharm = (pl.read_csv('data/raw/pharmacies_latlon14.csv')
      .select(pl.all().name.to_lowercase())
      .to_pandas()
)

gdf_pharm = gpd.GeoDataFrame(
    df_pharm, geometry=gpd.points_from_xy(df_pharm.lon, df_pharm.lat), crs=gdf_counties.crs
)

gdf_pharm=gdf_pharm.sjoin(gdf_counties[['geometry', 'geoid']], predicate='within').reset_index(drop=True)
# Which type of pharmacy --> add it to pharmacy data
df_pharm_type = lf_all.select(pl.col("buyer_dea_no", 'buyer_bus_act')).unique().collect().to_pandas()
gdf_pharm=gdf_pharm.merge(df_pharm_type, on='buyer_dea_no')
gdf_pharm.to_parquet('data/pharmacies_latlon.pq')

# Data on pharmacies (https://github.com/wpinvestigative/arcos-api/blob/master/data/pharmacies_latlon14.csv)
list_of_buyers_in_appa = gdf_pharm[gdf_pharm['geoid'].isin(appa_count_list)]['buyer_dea_no'].to_list()

# Reduce dimensionality of ARCOS dataset
## Filter for Central Appalachian states
## Filter for 2009-2011
## Filter for pharmacies in the buyer data
## Add county ("geoid") identifier 

lf_pharm = pl.scan_parquet('data/pharmacies_latlon.pq').select(pl.col("buyer_dea_no", "geoid"))

years = [2009, 2010, 2011]
lf_appa = (pl.scan_parquet('data/raw/arcos_all_washpost.pq')
      .select(pl.col("reporter_dea_no", 'buyer_dea_no', 'buyer_bus_act', 'buyer_state', 'drug_code', 'drug_name', 'mme_conversion_factor', 'quantity', 'transaction_code', 'transaction_date', 'calc_base_wt_in_gm', 'dosage_unit', 'product_name', 'ingredient_name', 'revised_company_name', 'dos_str'))
      .filter(pl.col("buyer_state").is_in(states))
      .filter(pl.col("transaction_date").dt.year().is_in(years))
      .filter(pl.col("buyer_dea_no").is_in(list_of_buyers_in_appa))
      .join(lf_pharm, left_on='buyer_dea_no', right_on='buyer_dea_no')
)

lf_appa.sink_parquet('data/arcos_appa.pq')

years = [2008, 2009, 2010, 2011]
lf_flo = (pl.scan_parquet('data/raw/arcos_all_washpost.pq')
      .select(pl.col("reporter_dea_no", 'buyer_dea_no', 'buyer_bus_act', 'buyer_state', 'drug_code', 'drug_name', 'mme_conversion_factor', 'quantity', 'transaction_code', 'transaction_date', 'calc_base_wt_in_gm', 'dosage_unit', 'product_name', 'ingredient_name', 'revised_company_name', 'dos_str'))
      .filter(pl.col("buyer_state") == 'FL')
      .filter(pl.col("transaction_date").dt.year().is_in(years))
      .join(lf_pharm, left_on='buyer_dea_no', right_on='buyer_dea_no')
)
lf_flo.sink_parquet('data/arcos_flo.pq')