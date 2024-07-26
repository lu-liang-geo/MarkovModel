import pandas
import numpy
import matplotlib.pyplot as plt
import helper_utils
import geopandas
import datetime


def analyze_missing_data(df, data_timestep):
    print(data_timestep.capitalize() + " Data:")
    
    location_ids = df["Location ID"].unique()
    
    print("Number of locations: ", len(location_ids))
    
    location_to_days = {}
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        
        location_timesteps = len(location_rows)
        location_to_days[location_id] = location_timesteps
        print("Location ID", location_id, "has data for", location_timesteps, data_timestep + "s")
    print()
        
    #Sort locations by number of days with data so bar chart looks cleaner
    locations_sorted = sorted(location_to_days.items(), key=lambda item: item[1])
    location_strings = []
    num_timesteps = []
    
    locations_sorted.reverse()
    for item in locations_sorted:
        location_strings.append(str(item[0]))
        num_timesteps.append(item[1])
                        
    summary_df = pandas.DataFrame(num_timesteps)
    print(summary_df.describe())
    print("\n\n")
                        
    plt.figure(figsize=(12, 9))
    plt.bar(location_strings, num_timesteps, width=0.4, color="blue")
    plt.title("Number of " + data_timestep.capitalize() + "s With Data by Location ID, Dec. 2021 - Nov. 2022")
    plt.xlabel("Location ID")
    plt.xticks(rotation=90)
    plt.ylabel("Num " + data_timestep.capitalize() + "s With Data")
    plt.savefig("plots/Missing Data/Num" + data_timestep.capitalize() + "sWithData.png")


def analyze_missing_data_by_month(df, data_timestep):
    df["month"] = df["Timestamp"].str[5:7]    
    months = ["12/21", "01/22", "02/22", "03/22", "04/22", "05/22", "06/22",
                  "07/22", "08/22", "09/22", "10/22", "11/22"]
    locations_per_month = []
    for month in months:
        month_df = df[df["month"] == month[:2]]
        month_locations_df = month_df["Location ID"].unique()
        num_month_locations = len(month_locations_df)
        locations_per_month.append(num_month_locations)
        
    plt.figure(figsize=(12, 9))
    plt.bar(months, locations_per_month, width=0.4, color="blue")
    plt.title("Number of Locations by Month, " + data_timestep.capitalize() + " Data, Dec. 2021 - Nov. 2022")
    plt.xlabel("Month")
    plt.ylabel("Num Locations With Data")
    plt.savefig("plots/Missing Data/NumLocationsByMonth" + data_timestep.capitalize() + "Data.png")


def plot_location_completeness(df, data_timestep, season="overall"):
    season_df = helper_utils.filter_season(df, season)
    location_ids = season_df["Location ID"].unique()
    
    location_data = []
    for location_id in location_ids:
        location_rows = season_df[season_df["Location ID"] == location_id]
        num_location_timesteps = len(location_rows)
        
        #Scale down hourly points so they appear a reasonable size when plotted
        if data_timestep == "hour":
            num_location_timesteps *= 0.3
        
        data = [location_id, location_rows.iloc[0]["Latitude"], location_rows.iloc[0]["Longitude"], num_location_timesteps]
        location_data.append(data)
        
    location_df = pandas.DataFrame(location_data, columns=["location_id", "Latitude", "Longitude", "num_timesteps"])
    
    county_data = geopandas.read_file("Tx_Census_CntyGeneralCoast_TTU/Tx_Census_CntyGeneralCoast_TTU.shp")
    ax = county_data[county_data["NAME"] == "Denton County"].plot(color='lightblue', edgecolor='black')
    
    gdf = geopandas.GeoDataFrame(location_df, geometry=geopandas.points_from_xy(location_df["Longitude"], location_df["Latitude"]))
    gdf.plot(ax=ax, color="blue", edgecolor="black", markersize=location_df["num_timesteps"])
    plt.title(data_timestep.capitalize() + "s With Data By Location, " + season.capitalize())
    plt.savefig("Plots/Missing Data/Maps/Num" + data_timestep.capitalize() + "sByLocation" + season.capitalize() + ".png")
    
    
def analyze_largest_missing_chunks_daily(df):
    location_ids = df["Location ID"].unique()
    locations_to_max_missing_chunk = {}
    
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        prev_date = None
        max_missing_days = 0
        
        for index, row in location_rows.iterrows():
            curr_date = datetime.datetime.strptime(row["day"], "%Y-%m-%d")
            if prev_date != None:
                delta = curr_date - prev_date
                days = delta.total_seconds() / (60*60*24)
                
                max_missing_days = max(max_missing_days, days)
                
            prev_date = curr_date
        
        locations_to_max_missing_chunk[location_id] = max_missing_days
        
    
    #Sort locations by number of days with data so bar chart looks cleaner
    locations_sorted = sorted(locations_to_max_missing_chunk.items(), key=lambda item: item[1])
    location_strings = []
    missing_chunks = []
    
    locations_sorted.reverse()
    for item in locations_sorted:
        location_strings.append(str(item[0]))
        missing_chunks.append(item[1])
        
    summary_df = pandas.DataFrame(missing_chunks)
    print("Max Missing Days:")
    print(summary_df.describe())
    print("\n\n")
        
    plt.figure(figsize=(12, 9))
    plt.bar(location_strings, missing_chunks, width=0.4, color="blue")
    plt.title("Length of Longest Chunk of Missing Days by Location ID, Dec. 2021 - Nov. 2022")
    plt.xlabel("Location ID")
    plt.xticks(rotation=90)
    plt.ylabel("Num Consecutive Missing Days")
    plt.savefig("plots/Missing Data/MaxMissingDays.png")
    
    
def analyze_largest_missing_chunks_hourly(df):
    df["time"] = df["Timestamp"].str[:19]
    
    location_ids = df["Location ID"].unique()
    locations_to_max_missing_chunk = {}
    
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        prev_date = None
        max_missing_hours = 0
        
        for index, row in location_rows.iterrows():
            curr_date = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
            if prev_date != None:
                delta = curr_date - prev_date
                hours = delta.total_seconds() / (60*60)
                
                max_missing_hours = max(max_missing_hours, hours)
                
            prev_date = curr_date
        
        locations_to_max_missing_chunk[location_id] = max_missing_hours
        
    
    #Sort locations by number of days with data so bar chart looks cleaner
    locations_sorted = sorted(locations_to_max_missing_chunk.items(), key=lambda item: item[1])
    location_strings = []
    missing_chunks = []
    
    locations_sorted.reverse()
    for item in locations_sorted:
        location_strings.append(str(item[0]))
        missing_chunks.append(item[1])
        
    summary_df = pandas.DataFrame(missing_chunks)
    print("Max Missing Hours:")
    print(summary_df.describe())
    print("\n\n")
        
    plt.figure(figsize=(12, 9))
    plt.bar(location_strings, missing_chunks, width=0.4, color="blue")
    plt.title("Length of Longest Chunk of Missing Hours by Location ID, Dec. 2021 - Nov. 2022")
    plt.xlabel("Location ID")
    plt.xticks(rotation=90)
    plt.ylabel("Num Consecutive Missing Hours")
    plt.savefig("plots/Missing Data/MaxMissingHours.png")
    


daily_df = pandas.read_csv("data/PA_geotagged_daily.csv")
daily_df = helper_utils.filter_time(daily_df)
# analyze_missing_data(daily_df, "day")
# analyze_missing_data_by_month(daily_df, "day")
analyze_largest_missing_chunks_daily(daily_df)

hourly_df = pandas.read_csv("data/New PA Data/PA_geotagged_hourly_raw_filtered.csv")
hourly_df = helper_utils.filter_time(hourly_df)
# analyze_missing_data(hourly_df, "hour")
# analyze_missing_data_by_month(hourly_df, "hour")
analyze_largest_missing_chunks_hourly(hourly_df)

# plot_location_completeness(daily_df, "day")
# plot_location_completeness(daily_df, "day", "spring")
# plot_location_completeness(daily_df, "day", "summer")
# plot_location_completeness(daily_df, "day", "fall")
# plot_location_completeness(daily_df, "day", "winter")

# plot_location_completeness(hourly_df, "hour")
# plot_location_completeness(hourly_df, "hour", "spring")
# plot_location_completeness(hourly_df, "hour", "summer")
# plot_location_completeness(hourly_df, "hour", "fall")
# plot_location_completeness(hourly_df, "hour", "winter")
