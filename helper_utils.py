import datetime
import pandas
import sys
import geopandas
import os
import matplotlib.pyplot as plt
import map_utils

def filter_season(df, season):
    if season == "overall":
        return df
    
    season_months = {
        "spring": ["03", "04", "05"],
        "summer": ["06", "07", "08"],
        "winter": ["12", "01", "02"],
        "fall": ["09", "10", "11"]
    }

    if not season in season_months:
        raise ValueError(
            "Error: Expected 'season' to be 'winter', 'spring', 'summer' or 'fall' but not " + season)

    months = season_months[season]

    season_df = df
    season_df["month"] = season_df["day"].str[5:7]
    season_df = season_df[season_df["month"].isin(months)]

    # Drop column so returned result is same schema as original df
    return season_df.drop(columns=["month"])


def encode_date(row):
    return datetime.datetime.strptime(row["day"], "%Y-%m-%d").date()


def filter_time(df):
    df["day"] = df["Timestamp"].str[:10]
    df["date"] = df.apply(encode_date, axis=1)
    
    start_date = datetime.date.fromisoformat("2021-12-01")
    
    filtered_df = df[df["date"] > start_date]
    return filtered_df


#Filter out locations with too few timesteps relative to the max number of possible timesteps
def filter_locations_without_enough_data_hourly(df, season="overall", threshold=0.2):
    season_df = filter_season(df, season)
    location_ids = season_df["Location ID"].unique()
    
    max_timesteps_by_season = {
        "overall": 365 * 24,
        "spring": (31 + 30 + 31) * 24,
        "summer": (30 + 31 + 31) * 24,
        "fall": (30 + 31 + 30) * 24,
        "winter": (31 + 31 + 28) * 24 #Assume we aren't in a leap year
    }
    max_num_timesteps = max_timesteps_by_season[season]
    
    num_locations_removed = 0
    removed_dfs = []
    
    #Filter out locations
    for location_id in location_ids:
        location_rows = season_df.query("`Location ID` == @location_id")
        num_location_timesteps = len(location_rows)
        
        if num_location_timesteps <= max_num_timesteps * threshold:
            removed_dfs.append(location_rows)
            
            index = location_rows.index
            season_df.drop(index, inplace=True)
            num_locations_removed += 1
            
    #Make map of locations which are used for this season
    fig, ax = plt.subplots()
    
    map_df = season_df[["Location ID", "Latitude", "Longitude"]].drop_duplicates()
    map_gdf = geopandas.GeoDataFrame(map_df, geometry=geopandas.points_from_xy(
        map_df["Longitude"], map_df["Latitude"]))
    ax = map_utils.generate_axis()
    counties_data = geopandas.read_file('Tx_Census_CntyGeneralCoast_TTU/Tx_Census_CntyGeneralCoast_TTU.shp')
    counties_data[counties_data['NAME'] == "Denton County"].plot(color='lightblue', edgecolor='blue', ax=ax)

    map = map_gdf.plot(ax=ax, color='green', markersize=10)
    
    if len(removed_dfs) > 0:
        removed_df = pandas.concat(removed_dfs)
        removed_gdf = geopandas.GeoDataFrame(removed_df, geometry=geopandas.points_from_xy(
            removed_df["Longitude"], removed_df["Latitude"]))
        removed_gdf.plot(ax=ax, color='red', markersize=10)
    
    map_utils.make_map_pretty(map)
    ax.set_title(season.capitalize() + " Sensors")
    fig.savefig(os.path.join("plots", "Sensor Location Maps", season.capitalize() + ".png"))
            
    return season_df, num_locations_removed


def encode_categories(row):
    pm25_concentration = row["PM2.5 (ATM)"]

    # Encode air quality categories based on
    # https://www.epa.gov/sites/default/files/2016-04/documents/2012_aqi_factsheet.pdf
    if pm25_concentration <= 12:
        return "Good"
    elif pm25_concentration <= 35.4:
        return "Moderate"
    elif pm25_concentration <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25_concentration <= 150.4:
        return "Unhealthy"
    elif pm25_concentration <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def filter_daytime(df):
    df["hour"] = df["Timestamp"].str[11:13].astype(int)
    filtered_df = df[(df["hour"] >= 8) & (df["hour"] <= 18)]
    return filtered_df


def filter_nighttime(df):
    df["hour"] = df["Timestamp"].str[11:13].astype(int)
    filtered_df = df[(df["hour"] < 8) | (df["hour"] > 18)]
    return filtered_df


def divide_times_of_day(df):
    df["hour"] = df["Timestamp"].str[11:13].astype(int)
    
    morning_df = df[(df["hour"] > 5) & (df["hour"] <= 10)]
    midday_df = df[(df["hour"] > 10) & (df["hour"] <= 12+3)]
    afternoon_evening_df = df[(df["hour"] > 12+3) & (df["hour"] <= 12+9)]
    night_df = df[(df["hour"] <= 5) | (df["hour"] > 12+9)]
    
    return {
        "morning": morning_df,
        "midday": midday_df,
        "afternoonEvening": afternoon_evening_df,
        "night": night_df
    }


def encode_day_of_week(row):
    return datetime.datetime.strptime(row["day"], "%Y-%m-%d").date().weekday()


def filter_to_weekdays(df):
    df["day_of_week"] = df.apply(encode_day_of_week, axis=1)
    
    filtered_df = df[(df["day_of_week"] < 5)]   #Week starts with Monday=0
    return filtered_df


def filter_to_weekends(df):
    df["day_of_week"] = df.apply(encode_day_of_week, axis=1)
    
    filtered_df = df[(df["day_of_week"] >= 5)]   #Week starts with Monday=0
    return filtered_df


def obtain_continuous_chunks_daily(df):
    chunks = []
    
    df["day"] = df["Timestamp"].str[:10]
    df["pm25_state"] = df.apply(encode_categories, axis=1)
    location_ids = df["Location ID"].unique()
      
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        
        curr_chunk = []
        prev_date = None
        
        for index, row in location_rows.iterrows():
            curr_date = datetime.datetime.strptime(row["day"], "%Y-%m-%d")
            if prev_date != None:
                next_expected_date = prev_date + datetime.timedelta(days=1)
                if curr_date != next_expected_date:
                    chunks.append(curr_chunk)
                    curr_chunk = []
                    
            curr_chunk.append(row["pm25_state"])
            prev_date = curr_date

        if len(curr_chunk) > 0:
            chunks.append(curr_chunk)

    return chunks


def obtain_continuous_raw_chunks_daily(df):
    chunks = []
    
    df["day"] = df["Timestamp"].str[:10]
    location_ids = df["Location ID"].unique()
      
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        
        curr_chunk = []
        prev_date = None
        
        for index, row in location_rows.iterrows():
            curr_date = datetime.datetime.strptime(row["day"], "%Y-%m-%d")
            if prev_date != None:
                next_expected_date = prev_date + datetime.timedelta(days=1)
                if curr_date != next_expected_date:
                    chunks.append(curr_chunk)
                    curr_chunk = []
                    
            curr_chunk.append(row["PM2.5 (ATM)"])
            prev_date = curr_date

        if len(curr_chunk) > 0:
            chunks.append(curr_chunk)

    return chunks


def obtain_continuous_chunks_hourly(df):
    chunks = []
    
    df["time"] = df["Timestamp"].str[:19]
    df["pm25_state"] = df.apply(encode_categories, axis=1)
    location_ids = df["Location ID"].unique()
      
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        
        curr_chunk = []
        prev_time = None
        
        for index, row in location_rows.iterrows():
            curr_time = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
            if prev_time != None:
                next_expected_time = prev_time + datetime.timedelta(hours=1)
                if curr_time != next_expected_time:
                    chunks.append(curr_chunk)
                    curr_chunk = []
                    
            curr_chunk.append(row["pm25_state"])
            prev_time = curr_time

        if len(curr_chunk) > 0:
            chunks.append(curr_chunk)

    return chunks


#Helper functions adapted from Jacob's code in the UHI-UPI repo
def str_to_dt(dtStr, dtFormat="%Y-%m-%d %H:%M:%S"):
        return datetime.datetime.strptime(dtStr, dtFormat)
    
def get_date(dt):
    """ 
    Description:
        Return the date of timestamp object
    """

    return dt.replace(
        second=0, microsecond=0, minute=0, hour=0
    )

#Take hourly data and average it to the daily level
#This is used for day/night on the daily scale
def average_to_day(df):
    def avg_by_dt(df):
        """ 
        Description:
            Average the values recorded over each day
        """

        # Copy the dataframe so we don't make changes
        # to original
        #df = df.copy()

        df["time"] = df["Timestamp"].str[:19]
        df['time'] = df['time'].apply(str_to_dt)
        df['time'] = df['time'].apply(get_date)
        
        #df["Location ID"] = df["Location ID"].apply(str)
        
        location_ids = df["Location ID"].unique()
        new_dataframes = []
      
        for location_id in location_ids:
            location_rows = df[df["Location ID"] == location_id]

            # Group the datapoints by rounded datetime and average
            df_grpd = location_rows.groupby(['time'], as_index=False).mean()
            df_grpd["Timestamp"] = df_grpd["time"].apply(date_to_string)
            df_grpd["Location ID"] = location_id
            new_dataframes.append(df_grpd)

        new_df = pandas.concat(new_dataframes)
        new_df = new_df.sort_values(by=['Timestamp'])
        #print(new_df)
        #exit()
        return new_df

    def date_to_string(date):
        #print(date)
        #exit()
        return date.strftime("%Y-%m-%d %H:%M:%S")
    
    return avg_by_dt(df)


#Take hourly day or night data and remove sensor days without enough data
#Used for day/night daily scale analysis
def filter_incomplete_days(df):
    df["day"] = df["Timestamp"].str[:19]
    df['day'] = df['day'].apply(str_to_dt)
    df['day'] = df['day'].apply(get_date)
    
    new_dataframes = []
    
    location_ids = df["Location ID"].unique()
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        location_days = location_rows["day"].unique()
        
        for location_day in location_days:
            location_day_rows = location_rows[location_rows["day"] == location_day]
            
            if len(location_day_rows) >= 0.80 * 12:
                new_dataframes.append(location_day_rows)
    
    
    new_df = pandas.concat(new_dataframes)
    new_df = new_df.sort_values(by=['Timestamp'])
    return new_df


def impute_missing_state_daily(df):
    df["day"] = df["Timestamp"].str[:10]
    df["date"] = df.apply(encode_date, axis=1)
    
    df["pm25_state"] = df.apply(encode_categories, axis=1)
    
    start_date = datetime.date.fromisoformat("2021-12-01")
    location_ids = df["Location ID"].unique()
    
    new_dfs = []
    
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        base_entry = location_rows.head(1)
        
        for days in range(365):
            date = start_date + datetime.timedelta(days=days)
            day_entry = location_rows[location_rows["day"] == date.strftime("%Y-%m-%d")]
            
            if len(day_entry) == 0:
                day_entry = base_entry
                day_entry["day"] = date.strftime("%Y-%m-%d")
                day_entry["pm25_state"] = "Missing"
            
            new_dfs.append(day_entry)
            
    new_df = pandas.concat(new_dfs)
    return new_df


def impute_missing_values_daily(df):
    df["day"] = df["Timestamp"].str[:10]
    df["date"] = df.apply(encode_date, axis=1)
    
    start_date = datetime.date.fromisoformat("2021-12-01")
    location_ids = df["Location ID"].unique()
    
    new_dfs = []
    
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        base_entry = location_rows.head(1)
        
        for days in range(365):
            date = start_date + datetime.timedelta(days=days)
            day_entry = location_rows[location_rows["day"] == date.strftime("%Y-%m-%d")]
            
            if len(day_entry) == 0:
                day_entry = base_entry
                day_entry["day"] = date.strftime("%Y-%m-%d")
                day_entry["PM2.5 (ATM)"] = -1
            
            new_dfs.append(day_entry)
            
    new_df = pandas.concat(new_dfs)
    return new_df


def impute_missing_values_hourly(df):
    def encode_hour(row):
        return datetime.datetime.strptime(row["hour"], "%Y-%m-%d %H:%M:%S")
    
    df["hour"] = df["Timestamp"].str[:19]
    df["date"] = df.apply(encode_hour, axis=1)
    
    start_date = datetime.date.fromisoformat("2021-12-01")
    location_ids = df["Location ID"].unique()
    
    new_dfs = []
    
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        base_entry = location_rows.head(1)
        
        for hours in range(365 * 24):
            date = start_date + datetime.timedelta(hours=hours)
            hour_entry = location_rows[location_rows["hour"] == date.strftime("%Y-%m-%d %H:%M:%S")]
            
            if len(hour_entry) == 0:
                hour_entry = base_entry
                hour_entry["hour"] = date.strftime("%Y-%m-%d %H:%M:%S")
                hour_entry["PM2.5 (ATM)"] = -1
            
            new_dfs.append(hour_entry)
            
    new_df = pandas.concat(new_dfs)
    return new_df
