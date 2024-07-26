import pandas
import geopandas
import matplotlib.pyplot as plt
import map_utils
import os

fig, ax = plt.subplots()

df = pandas.read_csv("Data/PA_metadata.csv")
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(
        df["Location 1 Longitude"], df["Location 1 Latitude"]))

ax = map_utils.generate_axis()
counties_data = geopandas.read_file('Tx_Census_CntyGeneralCoast_TTU/Tx_Census_CntyGeneralCoast_TTU.shp')
counties_data[counties_data['NAME'] == "Denton County"].plot(color='lightblue', edgecolor='blue', ax=ax)

map = gdf.plot(ax=ax, color='black', markersize=10)
map_utils.make_map_pretty(map)
ax.set_title("Sensor Network")
fig.savefig(os.path.join("plots", "SensorNetwork.png"))