import pandas
import geopandas
import helper_utils
import matplotlib.pyplot as plt

daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
daily_df = helper_utils.filter_time(daily_df)
daily_df = daily_df[["PM2.5 (ATM)", "Sensor_ID", "Latitude", "Longitude"]]

mean_daily_df = daily_df.groupby("Sensor_ID").mean()
mean_daily_df["pm25_state"] = mean_daily_df.apply(helper_utils.encode_categories, axis=1)

geometry = geopandas.points_from_xy(mean_daily_df["Longitude"], mean_daily_df["Latitude"])
mean_daily_gdf = geopandas.GeoDataFrame(mean_daily_df, geometry=geometry, crs="EPSG:4326")

mean_daily_gdf.drop(columns=["Longitude", "Latitude"], inplace=True)

mean_daily_gdf.to_file("Area Map New/daily_mean.shp")