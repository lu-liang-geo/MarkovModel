import pandas
import helper_utils
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, HourLocator, DateFormatter


# Set MatplotLib formatting settings
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams["font.family"] = "Times New Roman"

pollutant = "PM2.5"
pollutant_column = pollutant + " (ATM)"
pollutant_formatted = "$" + pollutant[:2] + "_{" + pollutant[2:] + "}$"

mean_color = "red"


def plot_months():
    daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
    daily_df = helper_utils.filter_time(daily_df)
    daily_df["month"] = daily_df["day"].str[5:7]
    daily_df["month"] = daily_df["month"].astype(int) % 12

    temp_df = daily_df[["month", pollutant_column]]
    monthly_average_df = temp_df.groupby(by="month").mean()

    fig, ax = plt.subplots(dpi=400)

    monthly_average_df.plot(style=".-", linewidth=2,
                            markersize=8, ax=ax, legend=False)

    yearly_mean = daily_df[pollutant_column].mean()

    ax.hlines(yearly_mean, 0, 12, linestyles="dashed", colors=mean_color)
    ax.annotate("Yearly Mean", (6.7, 9), xytext=(6.2, 13), size=14, arrowprops=dict(
        arrowstyle="->", mutation_scale=28))
    
    max_y = monthly_average_df.max()
    
    march_x = 3
    ax.vlines(march_x, -1, max_y, linestyles="dashed", colors="black")
    
    june_x = march_x + 3
    ax.vlines(june_x, -1, max_y, linestyles="dashed", colors="black")
    
    september_x = june_x + 3
    ax.vlines(september_x, -1, max_y, linestyles="dashed", colors="black")
    
    ax.fill_betweenx(range(0, int(max_y.iloc[0]) + 1), 0, march_x, color='gray', alpha=0.5)
    ax.fill_betweenx(range(0, int(max_y.iloc[0]) + 1), june_x, september_x, color='gray', alpha=0.5)
    
    text_y = 16.3
    ax.text(1, text_y, "Winter", fontweight="bold", fontsize=10)
    ax.text(4, text_y, "Spring", fontweight="bold", fontsize=10)
    ax.text(7, text_y, "Summer", fontweight="bold", fontsize=10)
    ax.text(10, text_y, "Fall", fontweight="bold", fontsize=10)

    ax.set_xticks(range(0, 12, 1), labels=["December", "January", "February", "March", "April", "May",
                                           "June", "July", "August", "September", "October",
                                           "November"], rotation=45, weight="bold")

    ax.set_xlabel("Month", weight="bold")
    ax.set_ylabel("Average " + pollutant_formatted +
                  " Concentration ($\mu g/m^{3}$)", weight="bold")

    fig.savefig(os.path.join("plots", "MonthlyAveragePM2.5.png"),
                bbox_inches="tight", dpi=400)


def plot_days():
    daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
    daily_df = helper_utils.filter_time(daily_df)

    temp_df = daily_df[["day", pollutant_column]]

    start_date = datetime.datetime(2021, 12, 1)
    end_date = datetime.datetime(2022, 11, 30)

    curr_date = start_date
    new_df_list = []
    while curr_date < end_date:
        day_string = curr_date.strftime("%Y-%m-%d")

        day_df = temp_df[temp_df["day"] == day_string]
        if len(day_df) == 0:
            new_df = pandas.DataFrame(
                [[day_string, -1]], columns=["day", pollutant_column])
            new_df_list.append(new_df)

        curr_date += datetime.timedelta(days=1)

    new_df_list.append(temp_df)
    temp_df = pandas.concat(new_df_list)

    daily_average_df = temp_df.groupby(by="day").mean()
    daily_average_df.sort_index(inplace=True)

    fig, ax = plt.subplots(dpi=400)
    fig.set_size_inches(17, 6)

    daily_average_df.plot(style=".-", linewidth=1.75,
                          markersize=6, ax=ax, legend=False)

    yearly_mean = daily_df[pollutant_column].mean()

    ax.hlines(yearly_mean, 0, 365, linestyles="dashed", colors=mean_color)
    ax.annotate("Yearly Mean", (240, 9), xytext=(230, 18), size=14, arrowprops=dict(
        arrowstyle="->", mutation_scale=28))
    
    max_y = daily_average_df.max()
    
    march_x = 31 + 31 + 28 #December + January + February
    ax.vlines(march_x, -1, max_y, linestyles="dashed", colors="black")
    
    june_x = march_x + 31 + 30 + 31 #March + April + May
    ax.vlines(june_x, -1, max_y, linestyles="dashed", colors="black")
    
    september_x = june_x + 30 + 31 + 31 #June + July + August
    ax.vlines(september_x, -1, max_y, linestyles="dashed", colors="black")
    
    ax.fill_betweenx(range(0, int(max_y.iloc[0]) + 1), 0, march_x, color='gray', alpha=0.5)
    ax.fill_betweenx(range(0, int(max_y.iloc[0]) + 1), june_x, september_x, color='gray', alpha=0.5)

    text_y = 45
    ax.text(march_x / 2, text_y, "Winter", fontweight="bold", fontsize=12)
    ax.text(march_x + (june_x - march_x)/2, text_y, "Spring", fontweight="bold", fontsize=12)
    ax.text(june_x + (september_x - june_x)/2, text_y, "Summer", fontweight="bold", fontsize=12)
    ax.text(september_x + (september_x - june_x)/2, text_y, "Fall", fontweight="bold", fontsize=12)

    ax.set_xlabel("Day", weight="bold")
    ax.set_ylabel("Average " + pollutant_formatted +
                  " Concentration ($\mu g/m^{3}$)", weight="bold")

    fig.savefig(os.path.join("plots", "DailyAveragePM2.5.png"),
                bbox_inches="tight", dpi=400)


if __name__ == '__main__':
    plot_months()
    plot_days()
