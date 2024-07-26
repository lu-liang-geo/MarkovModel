import pandas
import helper_utils
import matplotlib.pyplot as plt
import numpy

plt.rcParams["font.family"] = "Times New Roman"

study_length_days = 365.0
hours_per_day = 24
study_length_hours = study_length_days * hours_per_day

def get_hourly_completeness():
    hourly_df = pandas.read_csv(
    "data/New PA Data/PA_geotagged_hourly_raw_filtered.csv")
    hourly_df = helper_utils.filter_time(hourly_df)
    hourly_location_ids = hourly_df["Location ID"].unique()

    hourly_location_to_completeness = {}
    for location_id in hourly_location_ids:
        location_rows = hourly_df.query("`Location ID` == @location_id")
        num_location_timesteps = len(location_rows)
        
        location_completeness = num_location_timesteps / study_length_hours * 100    
        hourly_location_to_completeness[location_id] = location_completeness
    
    return hourly_location_to_completeness


def get_daily_completeness():
    daily_df = pandas.read_csv(
    "data/New PA Data/PA_geotagged_daily.csv")
    daily_df = helper_utils.filter_time(daily_df)
    daily_df_location_ids = daily_df["Location ID"].unique()

    daily_location_to_completeness = {}
    for location_id in daily_df_location_ids:
        location_rows = daily_df[daily_df["Location ID"] == location_id]
        num_location_timesteps = len(location_rows)
        location_completeness = num_location_timesteps / 365 * 100    
        daily_location_to_completeness[location_id] = location_completeness
        
    return daily_location_to_completeness


def combine_daily_and_hourly(daily_location_to_completeness, hourly_location_to_completeness):
    all_locations = daily_location_to_completeness.keys() | hourly_location_to_completeness.keys()
    
    for location in all_locations:
        if location not in daily_location_to_completeness:
            daily_location_to_completeness[location] = 0
            
        if location not in hourly_location_to_completeness:
            hourly_location_to_completeness[location] = 0
            
    return all_locations, daily_location_to_completeness, hourly_location_to_completeness


daily_location_to_completeness = get_daily_completeness()
hourly_location_to_completeness = get_hourly_completeness()

all_locations, daily_location_to_completeness, hourly_location_to_completeness = combine_daily_and_hourly(daily_location_to_completeness, hourly_location_to_completeness)

#Sort so that largest hourly completeness values are to the left
sorted_hourly_dict = dict(sorted(hourly_location_to_completeness.items(), 
                                 key=lambda item: item[1], reverse=True))
locations_sorted = sorted_hourly_dict.keys()

daily_x_values = []
hourly_x_values = []

for location in locations_sorted:
    daily_x_values.append(daily_location_to_completeness[location])
    hourly_x_values.append(hourly_location_to_completeness[location])

fig, ax = plt.subplots(dpi=400, layout="constrained", figsize=(20, 8))

width = 0.32
x = numpy.arange(len(locations_sorted))
ax.bar(x, hourly_x_values, width, label="Hourly", edgecolor="black")
ax.bar(x + width, daily_x_values, width, label="Daily", edgecolor="black")

ax.set_xlabel("Location ID", fontsize=24, weight="bold")
ax.set_ylabel("Percentage Completeness (%)", fontsize=24, weight="bold")

ax.set_xticks(x, locations_sorted, rotation=90)
ax.set_yticks(numpy.arange(0, 100, 10))
ax.set_xlim(-0.5, len(locations_sorted) + 0.5)

ax.tick_params(axis='both', which='major', labelsize=20)

ax.legend(prop={'size': 20})
plt.savefig("plots/Missing Data/LocationPercentageCompleteness.png")

