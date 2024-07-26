import pandas
import helper_utils
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, HourLocator, DateFormatter


#Set MatplotLib formatting settings
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.rcParams["font.family"] = "Times New Roman"


pollutant = "PM2.5"
pollutant_column = pollutant + " (ATM)"
pollutant_formatted = "$" + pollutant[:2] + "_{" + pollutant[2:] + "}$"

if pollutant == "PM Difference":
    pollutant_column = pollutant
    pollutant_formatted = "$PM_{10} - PM_{2.5}$"


def adjust_hour(row):
    return (int(row["hour"]) - 10) % 24


def plot_average_hourly_concentrations_by_season(df, season="overall", combine_plots=False,
                                                 color="black", marker=".", suffix=""):
    if not combine_plots:
        plt.cla()    
    
    mean_df = helper_utils.filter_season(df, season)
    mean_df = mean_df[["Timestamp", pollutant_column]]
    mean_df["hour"] = mean_df["Timestamp"].str[11:13]
    mean_df["hour_adjusted"] = mean_df.apply(adjust_hour, axis=1)
    
    pm_label = season.capitalize()
    mean_df = mean_df.rename(columns={pollutant_column: pm_label})
        
    mean_df.groupby(["hour_adjusted"])[pm_label].mean().plot(style=".-", linewidth=2, markersize=8,
                                                             color=color, marker=marker)

    plt.xticks(range(0, 24, 1),labels=["10AM", "11AM", "12PM", "1PM", "2PM", "3PM", 
                                "4PM", "5PM", "6PM", "7PM", "8PM", "9PM", "10PM", "11PM", "12AM", "1AM", 
                                "2AM", "3AM", "4AM", "5AM", "6AM", "7AM", "8AM", "9AM"], rotation=45)    
    plt.xlabel("Hour", weight="bold")
    plt.ylabel("Average " +  pollutant_formatted + " Concentration ($\mu g/m^{3}$)", weight="bold")
    
    alpha = 0.5 #Drawing lines over the shading changes its intensity
                #So different alpha levels are needed for individual seasons vs. all seasons
    
    file = "Average" + pollutant + "ConcentrationByHour" + season.capitalize() + suffix + ".png"
    if combine_plots:
        file = "Average" + pollutant + "ConcentrationsByHourAllSeasons" + suffix + ".png"
        alpha = 0.1
        plt.legend(loc=(1.02, 0.6))
        
    min_y = 5
    max_y = 14
    plt.ylim(min_y, max_y)
    
    if pollutant == "PM Difference":
        min_y = 0
        max_y = 5
    
    plt.vlines([(5 - 10) % 24, 12+3 - 10, 12+9 - 10], min_y, max_y, linestyles='dashed', colors='black')
    plt.fill_betweenx(range(min_y, max_y), (10 - 10) % 24, 12+3-10, color='gray', alpha=alpha)
    plt.fill_betweenx(range(min_y, max_y), 12+9-10, (5 - 10) % 24, color='gray', alpha=alpha)
    
    text_y = 13.5
    plt.text((11 - 10), text_y, "Midday")
    plt.text((3.3 + 12 - 10) % 24, text_y, "Afternoon/Evening")
    plt.text((12 + 12 - 10), text_y, "Night")
    plt.text((6.5 - 10) % 24, text_y, "Morning")
        
    plt.gcf().set_dpi(400)
    
    plt.savefig(os.path.join("plots", "EDA", pollutant, file), bbox_inches="tight")


#This function will likely return a misleading plot for time periods with missing data
#For now, I'm just using complete enough data to get an idea of the trends
def plot_hourly_concentrations_for_days(df, location_id, start_day, num_days=3):
    week_df = df[["Timestamp", "PM2.5 (ATM)", "Location ID"]]
    week_df = week_df[week_df["Location ID"] == location_id]
    
    week_df["hour"] = week_df["Timestamp"].str[11:13]
    week_df["day"] = week_df["Timestamp"].str[:10]
    
    #Get all data for each day under consideration
    start_datetime = datetime.datetime.strptime(start_day, "%Y-%m-%d")
    day_dfs = []
    for i in range(num_days):
        next_day = start_datetime + datetime.timedelta(days=i)
        next_day_string = next_day.strftime("%Y-%m-%d")
        day_df = week_df[week_df["day"] == next_day_string]
        day_dfs.append(day_df)
        
    week_df = pandas.concat(day_dfs)
    week_df["time"] = week_df["Timestamp"].str[:16]
    
    pandas.set_option('display.max_rows', None)
    print(week_df)
    
    week_df.plot(x="time", y="PM2.5 (ATM)", kind="line", style=".-")
    
    #Get hour labels for every 8 hours
    times = week_df["time"].tolist()
    time_labels = []
    step = 8
    for i in range(len(times)):
        val = ""
        if i % step == 0:
            val = times[i][11:]
        
        time_labels.append(val)
        
    plt.xticks(ticks=range(len(time_labels)), labels=time_labels, rotation=45)
    plt.title("Hourly PM2.5 Concentration at Location " + location_id + " for\n" + str(num_days) + " Days Starting on " + start_day)
    plt.xlabel("Hour")
    plt.ylabel("PM2.5 Concentration")
    
    plt.savefig(os.path.join("plots", "EDA", "Location" + location_id + "_" + start_day + "_ConcentrationFor" + str(num_days) + "Days.png"))
    

hourly_df = pandas.read_csv(
        "data/Full PA Data/PA_geotagged_hourly_raw_filtered.csv")
hourly_df = helper_utils.filter_time(hourly_df)
hourly_df["PM Difference"] = hourly_df["PM10.0 (ATM)"] - hourly_df["PM2.5 (ATM)"]


season_colors = {
    "overall": "steelblue",
    "spring": "orange",
    "summer": "forestgreen",
    "fall": "darkred",
    "winter": "darkorchid"
}

season_grayscale_colors = {
    "overall": "0.0",
    "spring": "0.3",
    "summer": "0.4",
    "fall": "0.5",
    "winter": "0.6"
}

season_markers = {
    "overall": ".",
    "spring": "*",
    "summer": "d",
    "fall": "8",
    "winter": "P"
}


for season in ["overall", "spring", "summer", "fall", "winter"]:
    plot_average_hourly_concentrations_by_season(hourly_df, season, color=season_colors[season])
    
plt.cla()
for season in ["overall", "spring", "summer", "fall", "winter"]:
    plot_average_hourly_concentrations_by_season(hourly_df, season, combine_plots=True,
                                                 color=season_colors[season])

#Make grayscale plots
for season in ["overall", "spring", "summer", "fall", "winter"]:
    plot_average_hourly_concentrations_by_season(hourly_df, season, 
                                                 color="0.0", suffix="Grayscale")
    
plt.cla()
for season in ["overall", "spring", "summer", "fall", "winter"]:
    plot_average_hourly_concentrations_by_season(hourly_df, season, combine_plots=True,
                                                 color=season_grayscale_colors[season],
                                                 marker=season_markers[season],
                                                 suffix="Grayscale")

    
plot_hourly_concentrations_for_days(hourly_df, "2712", "2022-01-03")
plot_hourly_concentrations_for_days(hourly_df, "2712", "2022-01-03", num_days=1)
plot_hourly_concentrations_for_days(hourly_df, "2201", "2022-10-11")
plot_hourly_concentrations_for_days(hourly_df, "1818", "2022-05-05", num_days=7)