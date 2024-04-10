import polars as pl
import matplotlib.pyplot as plt
import geopandas as gpd
from data import county_demographics

def pull_meta_data(cols: list) -> dict:
    '''
    Pass a list of column names (list of strings) to get the data description.

    Args:
        cols (list): List of column/variable names. 

    Returns:
        dict: A dictionary with column names as keys and descriptions as values.
    '''
    # Read metadata
    df_meta = pl.read_csv('data/data_dictionary.csv')

    # Lower case column names 
    cols_meta = df_meta.with_columns(pl.col("ColumnName").str.to_lowercase()).filter(pl.col("ColumnName").is_in(cols))

    col_names = cols_meta.select(pl.col("ColumnName")).to_series().to_list()
    col_names_descrip = cols_meta.select(pl.col("Description")).to_series().to_list()

    cols_description={col: name for col,name in zip(col_names, col_names_descrip)}

    return cols_description

def fetch_county_demographics(states: list, states_fips: list, gdf_counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Pass a list of state names and FIPS-codes along with GeoDataFrame of relevant counties to fetch demographic info at the county level.

    Args:
        states (list): List of state names. 
        states_fips (list): List of state FIPS-codes. 

    Returns:
        GeoDataFrame: A GeoDataFrame with county demographics merged.
    '''
    states_fips_dict=dict(zip(states, states_fips))
    county_demo = pl.DataFrame(county_demographics.get_report()).with_columns(pl.col("State").replace(states_fips_dict).alias("statefp")).select(pl.col("County", "State", "statefp", "Income", "Population", "Housing", "Education")).unnest("Income", "Population", "Housing", "Education").rename({'Median Houseold Income': "Median Household Income"})
    county_demo=county_demo.filter(pl.col("State").is_in(states)).to_pandas()
    gdf_counties=gdf_counties.merge(county_demo, left_on=['namelsad', 'statefp'], right_on=['County', 'statefp'], how='left')
    
    return gdf_counties

def make_descrip_maps(gdf_counties: gpd.GeoDataFrame, gdf_states: gpd.GeoDataFrame, title='') -> plt.Figure:
    '''
    Pass two GeoDataFrames, one at the county level and one at the state level. 
    Note that the county GeoDataFrame should have info merged to it (see above).

    Args:
        gdf_counties (gpd.GeoDataFrame): County GeoDataFrame. 
        gdf_states (gpd.GeoDataFrame): State GeoDataFrame (only for state borders). 
        title (str): String to give the figure a fitting title.

    Returns:
        plt.Figure: A figure of county demographics (adjust variables if needed).
    '''

    fig,ax = plt.subplots(2,1, figsize=(15,9))

    gdf_counties.plot(ax=ax[0], color='grey', edgecolor='r')
    gdf_states.plot(ax=ax[0], color='none', edgecolor='k')
    gdf_counties.plot(column="Bachelor's Degree or Higher", ax=ax[0],
            cmap='viridis',
            legend=True,
            legend_kwds={'label': "Bachelor degree or higher (%)",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.35, 8.2)})

    gdf_counties.plot(ax=ax[1], color='none', edgecolor='r')
    gdf_states.plot(ax=ax[1], color='none', edgecolor='k')
    gdf_counties.plot(column='Median Household Income', ax=ax[1],
            cmap='viridis',
            legend=True,
            legend_kwds={'label': "Median household income in USD",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.35, 8.2)})

    fig.suptitle(f'{title}')
    fig.text(0.3, 0.15, f'Note: This data covers the period 2010-2019.', fontsize=10)

    fig.tight_layout()
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.close()
    return fig

## BASIC TEMPLATE FOR DOC-STRINGS ##
'''
DESCRIPTION

Args:
    X (type): 

Returns:
    Something
'''