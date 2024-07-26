import pandas
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy


hourly_df = pandas.read_csv("data/Full PA Data/PA_geotagged_hourly_raw_filtered.csv")
hourly_location_ids = hourly_df["Location ID"].unique()
hourly_df["year"] = hourly_df["Timestamp"].str[0:4]
hourly_df["month"] = hourly_df["Timestamp"].str[5:7]

curr_month = datetime.datetime(2020, 3, 1)
end_month = datetime.datetime(2023, 1, 1)

months = []
num_sensors = []
num_records = []
while curr_month < end_month:
	year = curr_month.strftime("%Y")
	month = curr_month.strftime("%m")
	filtered_df = hourly_df.query("year == @year and month == @month")

	month_formatted = f"{year}/{month}"
	months.append(month_formatted)

	month_sensors = len(filtered_df["Location ID"].unique())
	num_sensors.append(month_sensors)
 
	month_records = len(filtered_df)
	num_records.append(month_records)
	
	curr_month = curr_month + relativedelta(months = 1)
 
#Plot number of active sensors with data
 
fig, ax = plt.subplots(dpi=400, layout="constrained", figsize=(12, 9))

ax.bar(months, num_sensors, width=1.0, color="green", edgecolor="black", linewidth=3)
ax.set_xlabel("Month", fontsize=20, weight="bold")
ax.set_ylabel("Number of Active Sensors", fontsize=20, weight="bold")

ax.set_xticks(months, months, rotation=90, weight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.tick_params(axis='both', which='major', labelsize=16)

start_x = 21 - 0.5
end_x = 33 + 0.5
max_y = 55
ax.vlines([start_x, end_x], 0, max_y, colors="brown", linestyles="dotted", linewidth=3)
ax.fill_betweenx(range(0, max_y), start_x, end_x, color='gray', alpha=0.5)

fig.savefig("plots/Study Period Validation/SensorsByMonth.png")

#Plot number of records

fig, ax = plt.subplots(dpi=400, layout="constrained", figsize=(12, 9))

ax.bar(months, num_records, width=1.0, color="green", edgecolor="black", linewidth=3)
ax.set_xlabel("Month", fontsize=20, weight="bold")
ax.set_ylabel("Number of Data Records", fontsize=20, weight="bold")

ax.set_xticks(months, months, rotation=90, weight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.tick_params(axis='both', which='major', labelsize=16)

start_x = 21 - 0.5
end_x = 33 + 0.5
max_y = 31500
ax.vlines([start_x, end_x], 0, max_y, colors="brown", linestyles="dotted", linewidth=3)
ax.fill_betweenx(range(0, max_y), start_x, end_x, color='gray', alpha=0.5)

fig.savefig("plots/Study Period Validation/RecordsByMonth.png")