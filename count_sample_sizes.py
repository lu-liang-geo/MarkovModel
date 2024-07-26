import pandas
import helper_utils


def count_for_season_daily(df: pandas.DataFrame, season):
    season_df = helper_utils.filter_season(df, season)
        
    season_df["PM State"] = season_df.apply(helper_utils.encode_categories, axis=1)
    
    print(f"Daily sample size counts for {season}")
    print(season_df["PM State"].value_counts())
    print()
    
    
def count_for_season_hourly(df: pandas.DataFrame, season):
    season_df, num_locations_removed = helper_utils.filter_locations_without_enough_data_hourly(df, season)
    season_df["PM State"] = season_df.apply(helper_utils.encode_categories, axis=1)
    
    print(f"Hourly sample size counts for {season}")
    print(season_df["PM State"].value_counts())
    print()


daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
daily_df = helper_utils.filter_time(daily_df)

hourly_df = pandas.read_csv(
	"data/Full PA Data/PA_geotagged_hourly_raw_filtered.csv")
hourly_df = helper_utils.filter_time(hourly_df)

for season in ["overall", "spring", "summer", "fall", "winter"]:
    count_for_season_daily(daily_df, season)
    
for season in ["overall", "spring", "summer", "fall", "winter"]:
    count_for_season_hourly(hourly_df, season)
