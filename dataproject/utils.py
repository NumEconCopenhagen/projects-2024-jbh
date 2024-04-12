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

    fig,ax = plt.subplots(3,1, figsize=(18,12))

    gdf_counties.plot(ax=ax[0], color='none', edgecolor='grey')
    gdf_states.plot(ax=ax[0], color='none', edgecolor='k')
    gdf_counties.plot(column="Bachelor's Degree or Higher", ax=ax[0],
            cmap='viridis',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Bachelor degree or higher (%)",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.35, 8.5)})

    gdf_counties.plot(ax=ax[1], color='none', edgecolor='grey')
    gdf_states.plot(ax=ax[1], color='none', edgecolor='k')
    gdf_counties.plot(column='Median Household Income', ax=ax[1],
            cmap='viridis',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Median household income in USD",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.35, 8.5)})
    
    gdf_counties.plot(ax=ax[2], color='none', edgecolor='grey')
    gdf_states.plot(ax=ax[2], color='none', edgecolor='k')
    gdf_counties.plot(column='Homeownership Rate', ax=ax[2],
            cmap='viridis',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Homeownership rate (%)",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.35, 8.5)})

    fig.suptitle(f'{title}')
    fig.text(0.3, 0.1, f'Note: This data covers the period 2010-2019.', fontsize=10)

    fig.tight_layout()
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    plt.close()
    return fig

def make_descrip_maps_shipment(gdf_counties: gpd.GeoDataFrame, gdf_states: gpd.GeoDataFrame, title='') -> plt.Figure:
    '''
    Pass two GeoDataFrames, one at the county level and one at the state level. 
    Note that the county GeoDataFrame should have info merged to it (see above).

    Args:
        gdf_counties (gpd.GeoDataFrame): County GeoDataFrame. 
        gdf_states (gpd.GeoDataFrame): State GeoDataFrame (only for state borders). 
        title (str): String to give the figure a fitting title.

    Returns:
        plt.Figure: A figure of county shipments (adjust variables if needed).
    '''

    fig,ax = plt.subplots(3,1, figsize=(18,12))

    gdf_counties.plot(ax=ax[0], color='none', edgecolor='grey')
    gdf_states.plot(ax=ax[0], color='none', edgecolor='k')
    gdf_counties[gdf_counties['date_labels']=='2009'].plot(column="MME_p_cap", ax=ax[0],
            cmap='plasma',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Opioids (grams) shipped per county per capita (2009)",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.38, 8.8)})


    gdf_counties.plot(ax=ax[1], color='none', edgecolor='grey')
    gdf_states.plot(ax=ax[1], color='none', edgecolor='k')
    gdf_counties[gdf_counties['date_labels']=='2010'].plot(column="MME_p_cap", ax=ax[1],
            cmap='plasma',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Opioids (grams) shipped per county per capita (2010)",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.38, 8.8)})
    
    gdf_counties.plot(ax=ax[2], color='none', edgecolor='grey')
    gdf_states.plot(ax=ax[2], color='none', edgecolor='k')
    gdf_counties[gdf_counties['date_labels']=='2011'].plot(column="MME_p_cap", ax=ax[2],
            cmap='plasma',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Opioids (grams) shipped per county per capita (2011)",
                            'orientation': "horizontal",
                            "shrink":.2,
                            'anchor': (0.38, 8.8)})

    fig.suptitle(f'{title}')
    fig.text(0.3, 0.1, f'Note: Weight of opioids is scaled by the morphine equivalent factor, see data description.', fontsize=10)

    fig.tight_layout()
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    plt.close()
    return fig

def make_helper_quarter_col(repeat: int) -> pl.DataFrame:
    # Generate numbers 1 to 4
    numbers = list(range(1, 5))

    # Repeat the numbers
    repeated_numbers = numbers * repeat  # Repeat x times

    # Create a Polars DataFrame
    df = pl.DataFrame({
        'repeated_numbers': repeated_numbers
    }).cast(pl.String)

    return df


def make_descrip_maps_shipment_single(gdf_counties: gpd.GeoDataFrame, gdf_states: gpd.GeoDataFrame,title='') -> plt.Figure:
    '''
    Pass two GeoDataFrames, one at the county level and one at the state level. 
    Note that the county GeoDataFrame should have info merged to it (see above).

    Args:
        gdf_counties (gpd.GeoDataFrame): County GeoDataFrame. 
        gdf_states (gpd.GeoDataFrame): State GeoDataFrame (only for state borders). 
        title (str): String to give the figure a fitting title.

    Returns:
        plt.Figure: A figure of county shipments (adjust variables if needed).
    '''

    fig,ax = plt.subplots(figsize=(8,7))

    gdf_counties.plot(ax=ax, color='none', edgecolor='grey')
    gdf_states.plot(ax=ax, color='none', edgecolor='k')
    gdf_counties[gdf_counties['date_labels']=='2010'].plot(column="MME_p_cap", ax=ax,
            cmap='plasma',
            alpha=0.8,
            legend=True,
            legend_kwds={'label': "Opioids (grams) shipped per county per capita (2010)",
                            'orientation': "horizontal",
                            "shrink":.3,
                            'anchor': (0.2, 7)})
    

    fig.suptitle(f'{title}')
    fig.tight_layout()
    ax.set_axis_off()
    
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