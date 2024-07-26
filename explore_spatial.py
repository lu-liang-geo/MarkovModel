import matplotlib.pyplot as plt
import numpy as numpy
import pandas
import geopandas
import helper_utils

# import pykrige.kriging_tools as kt
# from pykrige.ok import OrdinaryKriging

import skgstat

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 400

def remove_sensors_outside_denton_county(df):
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(
        df["Longitude"], df["Latitude"]
    ))

    county_data = geopandas.read_file(
        "Tx_Census_CntyGeneralCoast_TTU/Tx_Census_CntyGeneralCoast_TTU.shp")
    denton_county_data = county_data[county_data["NAME"] == "Denton County"]

    new_gdf = gdf.sjoin(denton_county_data, how="inner",
                        predicate="intersects")

    return new_gdf[["Latitude", "Longitude", "PM2.5 (ATM)"]]


daily_df = pandas.read_csv(
    "data/New PA Data/PA_geotagged_daily.csv")
daily_df = helper_utils.filter_time(daily_df)
daily_df = remove_sensors_outside_denton_county(daily_df)

kriging_df = daily_df[["Latitude", "Longitude", "PM2.5 (ATM)"]]
kriging_gdf = geopandas.GeoDataFrame(kriging_df, geometry=geopandas.points_from_xy(
    kriging_df["Longitude"], kriging_df["Latitude"]), crs="EPSG:4326")

kriging_gdf = kriging_gdf.to_crs("EPSG:3081")
kriging_gdf["x"] = kriging_gdf.geometry.x
kriging_gdf["y"] = kriging_gdf.geometry.y

kriging_df = kriging_gdf[["x", "y", "PM2.5 (ATM)"]]#.sample(frac=0.2)

kriging_arr = kriging_df.to_numpy()
# OK = OrdinaryKriging(
#     kriging_arr[:, 0],
#     kriging_arr[:, 1],
#     kriging_arr[:, 2],
#     variogram_model="gaussian",
#     verbose=True,
#     enable_plotting=True
# )

coords = tuple(zip(kriging_arr[:, 0], kriging_arr[:, 1]))

V = skgstat.Variogram(coords, kriging_arr[:, 2], model="gaussian")
#print(V.describe())

fig = V.plot(hist=False)
fig.set_size_inches(8, 5)

axes = fig.axes

axes[0].set_xlabel("Distance Threshold (m)")
axes[0].set_ylabel("Semivariance")

for i in range(len(axes)):
    x_label = axes[i].get_xlabel()
    axes[i].set_xlabel(x_label, weight="bold", fontsize="16")
    for label in axes[i].get_xticklabels():
        label.set_fontsize(12)

    y_label = axes[i].get_ylabel()
    axes[i].set_ylabel(y_label, weight="bold", fontsize="16")
    for label in axes[i].get_yticklabels():
        label.set_fontsize(12)
        
plt.savefig("plots/VariogramDaily.png")

# OrdinaryKriging.ex

# print(OK.lags)

#Note: Denton County is roughly a 50km x 50km square
#Corner-to-corner distance is then about 70km

