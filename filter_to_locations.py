import pandas

df = pandas.read_csv("data/New PA Data/PA_geotagged_hourly_raw_filtered.csv")
location_ids = [ "15", "2201", "1818", "200", "309", "1600", "5203", "2420", 
                "1212", "7414", "9200", "3024", "1615" ]

filtered_df = df[df["Location ID"].isin(location_ids)]
filtered_df.to_csv("data/output/filtered_PA_locations.csv", index=False)
