import pandas
import giddy
import numpy
import matplotlib.pyplot as plt
import helper_utils
import os
import sys
import random
import argparse
import matplotlib
from markov_helpers import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


states = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
          "Unhealthy", "Very Unhealthy", "Hazardous"]
num_states = len(states)
base_output_dir = os.path.join("results", "classic_markov_raw_filtered")

artificial_removal_chunks_path = os.path.join(base_output_dir,
                "artificial_removal_chunks.txt")

artificial_removal_sensitivity_analysis_path = os.path.join(base_output_dir,
                "artificial_removal_sensitivity_analysis.txt")

artificial_removal_random_path = os.path.join(base_output_dir,
                "artificial_removal_random.txt")

weekend_weekday_path = os.path.join(base_output_dir, "weekend_weekday.txt")
seasonal_thresholds_path = os.path.join(base_output_dir, "seasonal_thresholds.txt")
seasonal_path = os.path.join(base_output_dir, "seasonal.txt")
seasonal_homogeneity_tests_adjusted_path = os.path.join(base_output_dir,
                "seasonal_homogeneity_tests_adjusted.txt")
seasonal_homogeneity_tests_path = os.path.join(base_output_dir,
                "seasonal_homogeneity_tests.txt")

day_night_daily_path = os.path.join(base_output_dir, "day_night_daily.txt")
day_night_hourly_path = os.path.join(base_output_dir, "day_night_hourly.txt")
time_of_day_hourly_path = os.path.join(base_output_dir, "times_of_day_hourly.txt")
time_of_day_hourly_adjusted_path = os.path.join(base_output_dir, "times_of_day_hourly_adjusted.txt")

#Make custom colormaps

color_array = plt.get_cmap('binary')(range(256))
color_array[:,-1] = numpy.linspace(0.0, 0.5, 256)

map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name='greys_alpha', colors=color_array)
plt.register_cmap(cmap=map_object)

reds_adjusted = plt.get_cmap('Reds')(range(256))
reds_adjusted[:1, :] = numpy.array([1, 1, 1, 1])
reds_white_color = matplotlib.colors.ListedColormap(colors=reds_adjusted, name="reds_white")
plt.register_cmap(cmap=reds_white_color)

plt.rcParams["font.family"] = "Times New Roman"


def run_classic_markov(df):
    df["pm25_state"] = df.apply(helper_utils.encode_categories, axis=1)

    states = []
    location_ids = df["Location ID"].unique()

    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]
        location_pm25 = location_rows["pm25_state"].values.flatten().tolist()

        states.append(location_pm25)

    classic_markov_model = giddy.markov.Markov(
        numpy.array(states), summary=False)

    print(classic_markov_model.classes, "\n")
    print(classic_markov_model.p)


def run_classic_markov_chunks(chunks, name, save=True, path="plots/Classic Markov Heatmaps/"):
    total_transition_count_matrix = numpy.zeros((num_states, num_states))

    for chunk in chunks:
        classic_markov_model = giddy.markov.Markov(
            numpy.array([chunk]), summary=False, classes=numpy.array(states))

        transition_count_matrix = classic_markov_model.transitions
        total_transition_count_matrix = numpy.add(
            total_transition_count_matrix, transition_count_matrix)
        
    numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    transition_prob_matrix = []
    transition_sample_sizes = []

    print("Transition Probability Matrix:")
    print(str(states) + " | Sample Size")
    for transition_count_row in total_transition_count_matrix:
        total_transitions = transition_count_row.sum()
        transition_probability_row = transition_count_row

        if total_transitions > 0:
            transition_probability_row = numpy.multiply(
                transition_probability_row, 1.0/total_transitions)

        print(str(transition_probability_row) + " | " + str(total_transitions))
        transition_prob_matrix.append(transition_probability_row.tolist())
        transition_sample_sizes.append(total_transitions)
    print()
    
    # print("Transition Count Matrix:")
    # print(states)
    # print(total_transition_count_matrix)
    # print()

    print("First Mean Passage Time:")
    
    try:
        print(giddy.ergodic.fmpt(transition_prob_matrix, fill_empty_classes=True))
    except:
        print("Matrix is nonsingular, cannot calculate FMPT")
    
    print()

    print("Steady State")
    print(giddy.ergodic.steady_state(
        transition_prob_matrix, fill_empty_classes=True))
    print()

    print("Sojourn Time")
    print(giddy.markov.sojourn_time(transition_prob_matrix))

    print()

    if not save:
        result = ClassicMarkovResult(
            total_transition_count_matrix, transition_prob_matrix, transition_sample_sizes)
        return result

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5.5)

    threshold = 150 #TODO: Validate, possibly change for daily/hourly
    
    size_per_entry = 1
    plot_size = size_per_entry * num_states
    
    #row = [0 for _k in range(plot_size)]
    transition_matrix_to_plot = []
    #transition_matrix_to_plot = [row for _k in range(plot_size)]
    #print(transition_matrix_to_plot, file=sys.stderr)
    overlay_matrix = []
    
    for _i in range(plot_size):
        transition_matrix_to_plot.append([])
        overlay_matrix.append([])
    
    for i in range(num_states):
        for j in range(num_states):
            val = transition_prob_matrix[i][j]
            
            for x in range(size_per_entry):
                for y in range(size_per_entry):
                    overlay_val = 0
                    regular_val = val
                    
                    if transition_sample_sizes[i] < threshold:
                        overlay_val = 1.0
                    
                    transition_matrix_to_plot[i*size_per_entry + x].append(regular_val)
                    overlay_matrix[i*size_per_entry + x].append(overlay_val)

    position = ax.get_position()
    y_offset = 0.13
    x_offset = 0 if name == "Overall Daily" else -.01
    
    position.y0 += y_offset
    position.y1 += y_offset
    position.x0 += x_offset
    position.x1 += x_offset
    ax.set_position(position)
    
    print("Transition Matrix to Plot:")
    print(transition_matrix_to_plot)
    
    print("Overlay Matrix:")
    print(overlay_matrix)
    

    im = ax.imshow(transition_matrix_to_plot, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    ax.imshow(overlay_matrix, cmap="greys_alpha" )
    
    #ax.set_title("Transition Probability Matrix for " + name, fontsize=14)
    
    abbreviated_states = ["G", "M", "USG", "U", "VU", "H"]
    
    ax.set_xticks(numpy.arange(len(abbreviated_states)),
                  labels=abbreviated_states, rotation=45, fontsize=25, weight="bold")
    ax.set_yticks(numpy.arange(len(abbreviated_states)), 
                  labels=abbreviated_states, fontsize=25, weight="bold")
    
    dx = 5/72
    offset = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(os.path.join(path, name + "NoTicksNoLegend.png"))#, 
                #bbox_inches=matplotlib.transforms.Bbox([[0.5, 0], [5, 5.5]]))
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    plt.savefig(os.path.join(path, name + "XTicksNoLegend.png"))#, bbox_inches='tight')
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(True)
    plt.savefig(os.path.join(path, name + "YTicksNoLegend.png"))#, bbox_inches='tight')
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    
    plt.savefig(os.path.join(path, name + "NoLegend.png"))#, bbox_inches='tight')
    
    axins = inset_axes(ax,
                    width="85%",  
                    height="5%",
                    loc='lower center',
                    borderpad=-7.6
                   )
    colorbar = fig.colorbar(im, cax=axins, orientation="horizontal")
    tick_labels = colorbar.ax.get_xticklabels()
    colorbar.ax.set_xticklabels(tick_labels, fontsize=23, weight="bold")
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(os.path.join(path, name + "NoTicks.png"))#, bbox_inches='tight')
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    plt.savefig(os.path.join(path, name + "XTicks.png"))#, bbox_inches='tight')
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(True)
    plt.savefig(os.path.join(path, name + "YTicks.png"))#, bbox_inches='tight')
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    
    position = ax.get_position()
    labeled_x_offset = 0.03
    
    position.x0 += labeled_x_offset - x_offset
    position.x1 += labeled_x_offset - x_offset
    ax.set_position(position)
    
    plt.savefig(os.path.join(path, name + ".png"))#, bbox_inches='tight')

    result = ClassicMarkovResult(
        total_transition_count_matrix, transition_prob_matrix, transition_sample_sizes)
    return result


def run_for_season_daily(df, season, save_to_seasonal=True):
    if save_to_seasonal:
        file = open(seasonal_path, "a")
        sys.stdout = file

    season_df = helper_utils.filter_season(df, season)
    season_chunks = helper_utils.obtain_continuous_chunks_daily(season_df)

    print("\n----" + season.capitalize() + ", Daily----")
    result = run_classic_markov_chunks(season_chunks, season.capitalize() + " Daily",
                                       path="plots/Classic Markov Heatmaps/Seasonal Daily")

    if save_to_seasonal:
        file.close()

    return result


def run_for_season_hourly(df, season, save_to_seasonal=True):
    if save_to_seasonal:
        file = open(seasonal_path, "a")
        sys.stdout = file

    season_df, num_locations_removed = helper_utils.filter_locations_without_enough_data_hourly(df, season)
    season_chunks = helper_utils.obtain_continuous_chunks_hourly(season_df)

    print("\n----" + season.capitalize() + ", Hourly----")
    print("Removed", num_locations_removed, "locations based on large missing chunks")
    result = run_classic_markov_chunks(season_chunks, season.capitalize() + " Hourly",
                                       path="plots/Classic Markov Heatmaps/Seasonal Hourly", save=save_to_seasonal)

    if save_to_seasonal:
        file.close()

    return result


def run_season_threshold(df, season, threshold):
    filtered_df = helper_utils.filter_locations_without_enough_data(
        df, season=season, threshold=threshold)
    chunks = helper_utils.obtain_continuous_chunks_daily(filtered_df)
    print("Filtered " + season.capitalize() + " Hourly, " +
          str(threshold * 100) + "% Threshold")
    print()
    results = run_classic_markov_chunks(chunks, "Filtered " + season.capitalize() + " Hourly, " + str(threshold * 100) + "% Threshold",
                                        path="plots/Classic Markov Heatmaps/Filtered")
    print("---------------\n\n")

    return results


def perform_threshold_hypothesis_tests(hourly, hourly_25, hourly_50, season):
    matrix_list = [hourly, hourly_25]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["0%", "25%"])
    print(season.capitalize() + ", 0% vs. 25%")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [hourly, hourly_50]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["0%", "50%"])
    print(season.capitalize() + ", 0% vs. 50%")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [hourly_25, hourly_50]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["25%", "50%"])
    print(season.capitalize() + ", 25% vs. 50%")
    print(homogeneity_results.summary())
    print("---------------\n\n")


def run_all_season_thresholds(df, overall_hourly_t, spring_hourly_t, summer_hourly_t, fall_hourly_t, winter_hourly_t):
    file = open(seasonal_thresholds_path, "a")
    sys.stdout = file

    # Get all threshold results

    overall_hourly_25_result = run_season_threshold(df, "overall", 0.25)
    overall_hourly_50_result = run_season_threshold(df, "overall", 0.50)

    spring_hourly_25_result = run_season_threshold(df, "spring", 0.25)
    spring_hourly_50_result = run_season_threshold(df, "spring", 0.50)

    summer_hourly_25_result = run_season_threshold(df, "summer", 0.25)
    summer_hourly_50_result = run_season_threshold(df, "summer", 0.50)

    fall_hourly_25_result = run_season_threshold(df, "fall", 0.25)
    fall_hourly_50_result = run_season_threshold(df, "fall", 0.50)

    winter_hourly_25_result = run_season_threshold(df, "winter", 0.25)
    winter_hourly_50_result = run_season_threshold(df, "winter", 0.50)

    # Perform hypothesis tests to determine whether thresholds matter

    perform_threshold_hypothesis_tests(overall_hourly_t, overall_hourly_25_result.transition_count_matrix,
                                       overall_hourly_50_result.transition_count_matrix, "overall")
    perform_threshold_hypothesis_tests(spring_hourly_t, spring_hourly_25_result.transition_count_matrix,
                                       spring_hourly_50_result.transition_count_matrix, "spring")
    perform_threshold_hypothesis_tests(summer_hourly_t, summer_hourly_25_result.transition_count_matrix,
                                       summer_hourly_50_result.transition_count_matrix, "summer")
    perform_threshold_hypothesis_tests(fall_hourly_t, fall_hourly_25_result.transition_count_matrix,
                                       fall_hourly_50_result.transition_count_matrix, "fall")
    perform_threshold_hypothesis_tests(winter_hourly_t, winter_hourly_25_result.transition_count_matrix,
                                       winter_hourly_50_result.transition_count_matrix, "winter")

    file.close()


def perform_season_homogeneity_tests_adjusted(spring_result, summer_result, fall_result, winter_result, timescale: str):
    file = open(seasonal_homogeneity_tests_adjusted_path, "a")
    sys.stdout = file

    print(timescale + " Seasonal Homogeneity Test Adjusted")

    matrix_list = generate_adjusted_matrix_list(
        [spring_result, summer_result, fall_result, winter_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Summer", "Fall", "Winter"],
                                                   title="Hourly Seasonal Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list(
        [spring_result, fall_result, winter_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Fall", "Winter"],
                                                   title="Hourly Seasonal Homogeneity Test, No Summer")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list([spring_result, fall_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Fall"],
                                                   title="Hourly Spring/Fall Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list([spring_result, winter_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Winter"],
                                                   title="Hourly Spring/Winter Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list([winter_result, fall_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Winter", "Fall"],
                                                   title="Hourly Winter/Fall Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list([fall_result, summer_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Fall", "Summer"],
                                                   title="Hourly Fall/Summer Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list([spring_result, summer_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Summer"],
                                                   title="Hourly Spring/Summer Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = generate_adjusted_matrix_list([winter_result, summer_result])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Winter", "Summer"],
                                                   title="Hourly Winter/Summer Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    file.close()


def perform_season_homogeneity_tests(spring_result, summer_result, fall_result, winter_result, timescale: str):
    file = open(seasonal_homogeneity_tests_path, "a")
    sys.stdout = file

    print(timescale + " Seasonal Homogeneity Test")

    matrix_list = [spring_result.transition_count_matrix, summer_result.transition_count_matrix, 
                   fall_result.transition_count_matrix, winter_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Summer", "Fall", "Winter"],
                                                   title="Hourly Seasonal Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [spring_result.transition_count_matrix, fall_result.transition_count_matrix, 
                   winter_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Fall", "Winter"],
                                                   title="Hourly Seasonal Homogeneity Test, No Summer")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [spring_result.transition_count_matrix, fall_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Fall"],
                                                   title="Hourly Spring/Fall Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [spring_result.transition_count_matrix, winter_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Winter"],
                                                   title="Hourly Spring/Winter Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [winter_result.transition_count_matrix, fall_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Winter", "Fall"],
                                                   title="Hourly Winter/Fall Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [fall_result.transition_count_matrix, summer_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Fall", "Summer"],
                                                   title="Hourly Fall/Summer Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [spring_result.transition_count_matrix, summer_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Spring", "Summer"],
                                                   title="Hourly Spring/Summer Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    matrix_list = [winter_result.transition_count_matrix, summer_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Winter", "Summer"],
                                                   title="Hourly Winter/Summer Homogeneity Test")
    print(homogeneity_results.summary())
    print("---------------\n\n")

    file.close()


def run_weekend_weekday_hourly(df):
    file = open(weekend_weekday_path, "a")
    sys.stdout = file

    weekday_hourly_df = helper_utils.filter_to_weekdays(df)
    chunks = helper_utils.obtain_continuous_chunks_hourly(weekday_hourly_df)
    print("Weekdays Overall Hourly")
    print()
    weekdays_hourly_result = run_classic_markov_chunks(
        chunks, "Weekdays Overall Hourly")

    weekend_hourly_df = helper_utils.filter_to_weekends(df)
    chunks = helper_utils.obtain_continuous_chunks_hourly(weekend_hourly_df)
    print("Weekends Overall Hourly")
    print()
    weekends_hourly_result = run_classic_markov_chunks(
        chunks, "Weekends Overall Hourly")

    matrix_list = [weekdays_hourly_result.transition_count_matrix,
                   weekends_hourly_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Weekdays", "Weekends"],
                                                   title="Overall Hourly Weekday/Weekend Homogeneity Test")
    print(homogeneity_results.summary())

    file.close()


def run_weekend_weekday_daily(df):
    file = open(weekend_weekday_path, "a")
    sys.stdout = file

    weekday_daily_df = helper_utils.filter_to_weekdays(df)
    chunks = helper_utils.obtain_continuous_chunks_daily(weekday_daily_df)
    print("Weekdays Overall Daily")
    print()
    weekdays_hourly_result = run_classic_markov_chunks(
        chunks, "Weekdays Overall Daily")

    weekday_daily_df = helper_utils.filter_to_weekends(df)
    chunks = helper_utils.obtain_continuous_chunks_daily(weekday_daily_df)
    print("Weekends Overall Daily")
    print()
    weekends_hourly_result = run_classic_markov_chunks(
        chunks, "Weekends Overall Daily")

    matrix_list = [weekdays_hourly_result.transition_count_matrix,
                   weekends_hourly_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=[
                                                       "Weekdays", "Weekends"],
                                                   title="Overall Daily Weekday/Weekend Homogeneity Test")
    print(homogeneity_results.summary())

    file.close()


def run_day_night_daily(df, season="overall"):
    file = open(day_night_daily_path, "a")
    sys.stdout = file

    daytime_df = helper_utils.filter_daytime(df)
    daytime_df = helper_utils.filter_season(daytime_df, season)
    
    #Hourly data is passed in to filter to day/night
    #Now must aggregate to the daily scale before chunking
    daytime_df = helper_utils.filter_incomplete_days(daytime_df)
    daytime_df = helper_utils.average_to_day(daytime_df)
    chunks = helper_utils.obtain_continuous_chunks_daily(daytime_df)
        
    print("Daytime " + season.capitalize())
    print()
    daytime_result = run_classic_markov_chunks(
        chunks, "Daytime " + season.capitalize() + " Daily",
        path="plots/Classic Markov Heatmaps/Day Night")

    nighttime_df = helper_utils.filter_nighttime(df)
    nighttime_df = helper_utils.filter_season(nighttime_df, season)
    
    #Hourly data is passed in to filter to day/night
    #Now must aggregate to the daily scale before chunking
    nighttime_df = helper_utils.filter_incomplete_days(nighttime_df)
    nighttime_df = helper_utils.average_to_day(nighttime_df)
    chunks = helper_utils.obtain_continuous_chunks_daily(nighttime_df)
    
    print("Nighttime " + season.capitalize())
    print()
    nighttime_result = run_classic_markov_chunks(
        chunks, "Nighttime " + season.capitalize() + " Daily",
        path="plots/Classic Markov Heatmaps/Day Night")

    matrix_list = [daytime_result.transition_count_matrix,
                   nighttime_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["Daytime", "Nighttime"])
    print(homogeneity_results.summary())

    file.close()


def run_day_night_hourly(df, season="overall"):
    file = open(day_night_hourly_path, "a")
    sys.stdout = file

    daytime_df = helper_utils.filter_daytime(df)
    daytime_df = helper_utils.filter_season(daytime_df, season)
    chunks = helper_utils.obtain_continuous_chunks_hourly(daytime_df)
    
    print("Daytime " + season.capitalize())
    print()
    daytime_result = run_classic_markov_chunks(
        chunks, "Daytime " + season.capitalize() + " Hourly",
        path="plots/Classic Markov Heatmaps/Day Night")

    nighttime_df = helper_utils.filter_nighttime(df)
    nighttime_df = helper_utils.filter_season(nighttime_df, season)
    chunks = helper_utils.obtain_continuous_chunks_hourly(nighttime_df)
    print("Nighttime " + season.capitalize())
    print()
    nighttime_result = run_classic_markov_chunks(
        chunks, "Nighttime " + season.capitalize() + " Hourly",
        path="plots/Classic Markov Heatmaps/Day Night")

    matrix_list = [daytime_result.transition_count_matrix,
                   nighttime_result.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["Daytime", "Nighttime"])
    print(homogeneity_results.summary())

    file.close()


def test_artificial_removal_chunks(df, run_func):
    file = open(artificial_removal_chunks_path, "a")
    sys.stdout = file

    # Filter to locations with lots of data
    df = helper_utils.filter_locations_without_enough_data(df, threshold=0.80)

    # Run classic Markov on those locations
    print("Locations with > 80% Completeness")
    overall_result = run_func(df, "overall", save_to_seasonal=False)

    # Remove some chunks of data from locations
    location_ids = df["Location ID"].unique()

    new_dataframes = []
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]

        num_location_timesteps = len(location_rows)
        portion_to_remove = random.uniform(0.4, 0.7)
        num_to_remove = int(num_location_timesteps * portion_to_remove)

        location_rows.drop(
            index=location_rows.index[:num_to_remove], inplace=True)
        new_dataframes.append(location_rows)

    new_df = pandas.concat(new_dataframes)

    # Run classic Markov again
    print("Locations with > 80% Completeness, Chunks Removed")
    overall_result_data_removed = run_func(
        new_df, "overall", save_to_seasonal=False)

    # Check if results changed
    matrix_list = [overall_result.transition_count_matrix,
                   overall_result_data_removed.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["80%", "80%, Chunks Removed"])
    print("Raw Hypothesis Test")
    print(homogeneity_results.summary())

    matrix_list = generate_adjusted_matrix_list(
        [overall_result, overall_result_data_removed])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["80%", "80%, Chunks Removed"])
    print("\nSample Size Adjusted Hypothesis Test")
    print(homogeneity_results.summary())

    file.close()


def test_artificial_removal_random(df, run_func):
    file = open(artificial_removal_random_path, "a")
    sys.stdout = file

    # Filter to locations with lots of data
    df = helper_utils.filter_locations_without_enough_data(df, threshold=0.80)

    # Run classic Markov on those locations
    print("Locations with > 80% Completeness")
    overall_result = run_func(df, "overall", save_to_seasonal=False)

    # Remove random rows of data from locations
    location_ids = df["Location ID"].unique()

    new_dataframes = []
    for location_id in location_ids:
        location_rows = df[df["Location ID"] == location_id]

        num_location_timesteps = len(location_rows)
        portion_to_remove = random.uniform(0.4, 0.7)
        num_to_remove = int(num_location_timesteps * portion_to_remove)

        drop_indices = numpy.random.choice(
            location_rows.index, num_to_remove, replace=False)
        location_rows.drop(drop_indices, inplace=True)
        new_dataframes.append(location_rows)

    new_df = pandas.concat(new_dataframes)

    # Run classic Markov again
    print("Locations with > 80% Completeness, Random Rows Removed")
    overall_result_data_removed = run_func(
        new_df, "overall", save_to_seasonal=False)

    # Check if results changed
    matrix_list = [overall_result.transition_count_matrix,
                   overall_result_data_removed.transition_count_matrix]
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["80%", "80%, Random Rows Removed"])
    print("Raw Hypothesis Test")
    print(homogeneity_results.summary())

    matrix_list = generate_adjusted_matrix_list(
        [overall_result, overall_result_data_removed])
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=["80%", "80%, Random Rows Removed"])
    print("\nSample Size Adjusted Hypothesis Test")
    print(homogeneity_results.summary())

    file.close()


def perform_artificial_chunk_sensitivity_analysis(df, run_func):
    file = open(artificial_removal_sensitivity_analysis_path, "a")
    sys.stdout = file

    # Filter to locations with lots of data
    df = helper_utils.filter_locations_without_enough_data(df, threshold=0.80)

    # Run classic Markov on those locations
    print("Locations with > 80% Completeness")
    overall_result = run_func(df, "overall", save_to_seasonal=False)

    # Remove some chunks of data from locations
    location_ids = df["Location ID"].unique()

    for missing_chunk_size in numpy.arange(0.1, 0.8, 0.1):
        print("Missing chunk size:", missing_chunk_size)
        new_dataframes = []
        for location_id in location_ids:
            location_rows = df[df["Location ID"] == location_id]

            num_location_timesteps = len(location_rows)
            num_to_remove = int(num_location_timesteps * missing_chunk_size)

            location_rows.drop(
                index=location_rows.index[:num_to_remove], inplace=True)
            new_dataframes.append(location_rows)

        new_df = pandas.concat(new_dataframes)

        # Run classic Markov again
        print("Locations with > 80% Completeness, Chunks Removed")
        overall_result_data_removed = run_func(
            new_df, "overall", save_to_seasonal=False)

        # Check if results changed
        matrix_list = [overall_result.transition_count_matrix,
                    overall_result_data_removed.transition_count_matrix]
        homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                    regime_names=["80%", "80%, Chunks Removed"])
        print("Raw Hypothesis Test")
        print(homogeneity_results.summary())

        matrix_list = generate_adjusted_matrix_list(
            [overall_result, overall_result_data_removed])
        homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                    regime_names=["80%", "80%, Chunks Removed"])
        print("\nSample Size Adjusted Hypothesis Test")
        print(homogeneity_results.summary())

    file.close()


def run_times_of_day_hourly(df, season="overall"):
    file = open(time_of_day_hourly_path, "a")
    sys.stdout = file
    
    season_df = helper_utils.filter_season(df, season)
    times_of_day = helper_utils.divide_times_of_day(season_df)
    
    matrix_list = []
    for time_of_day in times_of_day:
        time_of_day_df = times_of_day[time_of_day]
        print(time_of_day, season.capitalize(), "\n")
        
        chunks = helper_utils.obtain_continuous_chunks_hourly(time_of_day_df)
        result = run_classic_markov_chunks(
            chunks, time_of_day + season.capitalize() + " Hourly", 
            path="plots/Classic Markov Heatmaps/Times of Day")
        matrix_list.append(result.transition_count_matrix)
        
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=list(times_of_day.keys()))
    print(homogeneity_results.summary())
        
    file.close()


def run_times_of_day_hourly_adjusted(df, season="overall"):
    file = open(time_of_day_hourly_adjusted_path, "a")
    sys.stdout = file
    
    season_df = helper_utils.filter_season(df, season)
    times_of_day = helper_utils.divide_times_of_day(season_df)
    
    matrix_list = []
    results_list = []
    for time_of_day in times_of_day:
        time_of_day_df = times_of_day[time_of_day]
        print(time_of_day, season.capitalize(), "\n")
        
        chunks = helper_utils.obtain_continuous_chunks_hourly(time_of_day_df)
        result = run_classic_markov_chunks(
            chunks, time_of_day + season.capitalize() + " Hourly",
            path="plots/Classic Markov Heatmaps/Times of Day")
        matrix_list.append(result.transition_count_matrix)
        results_list.append(result)
        
        adjusted_prob_matrix = remove_low_sample_size_states(result)
        print("Adjusted Steady State: ")
        print(giddy.ergodic.steady_state(adjusted_prob_matrix))
        print()
        
    print("Regular Hypothesis Test:")
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=list(times_of_day.keys()))
    print(homogeneity_results.summary())
    
    print("Sample Size Adjusted Hypothesis Test:")
    matrix_list = generate_adjusted_matrix_list(results_list)
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=states,
                                                   regime_names=list(times_of_day.keys()))
    print(homogeneity_results.summary())
        
    file.close()


def run_classic_markov_imputed(df):
    #Note: 'day' column is already present and should be used 
    #Due to how imputation is done
    imputed_df = helper_utils.impute_missing_state_daily(df)
    missing_index = 1
    
    states = []
    location_ids = imputed_df["Location ID"].unique()

    for location_id in location_ids:
        location_rows = imputed_df[imputed_df["Location ID"] == location_id]
        location_pm25 = location_rows["pm25_state"].values.flatten().tolist()

        states.append(location_pm25)

    classic_markov_model = giddy.markov.Markov(
        numpy.array(states), summary=False)
        
    #Remove imputed MISSING state
    new_transition_counts = []
    for row in classic_markov_model.transitions:
        transition_count_row = numpy.delete(row, missing_index)
        new_transition_counts.append(transition_count_row)
        
    del new_transition_counts[missing_index]
    
    #Adjust transition matrix
    new_transition_probabilities = []
    for transition_count_row in new_transition_counts:
        total_transitions = transition_count_row.sum()
        transition_probability_row = numpy.multiply(
            transition_count_row, 1.0/total_transitions)
        
        new_transition_probabilities.append(transition_probability_row)
        
    return new_transition_probabilities


if __name__ == '__main__':
    # Parse arguments
    # Running all analyses every time takes a while
    # This provides flexibility in what to run

    parser = argparse.ArgumentParser(
        description='Run classic Markov analyses for Denton PM_2.5')

    parser.add_argument('--run_seasonal', action='store_true')
    parser.add_argument('--run_season_thresholds',
                        help="Test seasonal removal thresholds; will run seasonal", action='store_true')
    parser.add_argument('--run_day_night', action='store_true')
    parser.add_argument('--run_weekend_weekday', action='store_true')
    parser.add_argument('--run_artificial_removal', action='store_true')
    parser.add_argument('--run_times_of_day', action='store_true')
    parser.add_argument('--all', action='store_true', 
                        help="Specifies that all analyses should be run; takes precedence over other flags")

    args = parser.parse_args()
    
    #Might be needed for different analyses
    daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
    daily_df = helper_utils.filter_time(daily_df)
    
    hourly_df = pandas.read_csv(
        "data/Full PA Data/PA_geotagged_hourly_raw_filtered.csv")
    hourly_df = helper_utils.filter_time(hourly_df)

    if args.run_seasonal or args.run_season_thresholds or args.all:
        if os.path.exists(seasonal_path):
            os.remove(seasonal_path)
        if os.path.exists(seasonal_homogeneity_tests_adjusted_path):
            os.remove(seasonal_homogeneity_tests_adjusted_path)
        if os.path.exists(seasonal_homogeneity_tests_path):
            os.remove(seasonal_homogeneity_tests_path)
        
        # Do daily stuff
        run_for_season_daily(daily_df, "overall")
        spring_daily_result = run_for_season_daily(daily_df, "spring")
        summer_daily_result = run_for_season_daily(daily_df, "summer")
        fall_daily_result = run_for_season_daily(daily_df, "fall")
        winter_daily_result = run_for_season_daily(daily_df, "winter")

        perform_season_homogeneity_tests(spring_daily_result, summer_daily_result,
                                        fall_daily_result, winter_daily_result, "Daily")
        perform_season_homogeneity_tests_adjusted(spring_daily_result, summer_daily_result,
                                        fall_daily_result, winter_daily_result, "Daily")

        # Do hourly stuff
        overall_hourly_result = run_for_season_hourly(hourly_df, "overall")

        spring_hourly_result = run_for_season_hourly(hourly_df, "spring")
        summer_hourly_result = run_for_season_hourly(hourly_df, "summer")
        fall_hourly_result = run_for_season_hourly(hourly_df, "fall")
        winter_hourly_result = run_for_season_hourly(hourly_df, "winter")

        perform_season_homogeneity_tests(spring_hourly_result, summer_hourly_result,
                                        fall_hourly_result, winter_hourly_result, "Hourly")
        perform_season_homogeneity_tests_adjusted(spring_hourly_result, summer_hourly_result,
                                        fall_hourly_result, winter_hourly_result, "Hourly")

    # if args.run_day_night or args.all:
    #     if os.path.exists(day_night_daily_path):
    #         os.remove(day_night_daily_path)
    #     if os.path.exists(day_night_hourly_path):
    #         os.remove(day_night_hourly_path)
        
    #     for season in ["overall", "spring", "summer", "fall", "winter"]:
    #         run_day_night_daily(hourly_df, season) #Still needs hourly to filter to day/night
    #         run_day_night_hourly(hourly_df, season)

    # if args.run_season_thresholds or args.all:
    #     if os.path.exists(seasonal_thresholds_path):
    #         os.remove(seasonal_thresholds_path)
            
    #     run_all_season_thresholds(hourly_df, overall_hourly_result.transition_count_matrix,
    #                             spring_hourly_result.transition_count_matrix, summer_hourly_result.transition_count_matrix,
    #                             fall_hourly_result.transition_count_matrix, winter_hourly_result.transition_count_matrix)
        
    # if args.run_weekend_weekday or args.all:
    #     if os.path.exists(weekend_weekday_path):
    #         os.remove(weekend_weekday_path)
        
    #     run_weekend_weekday_hourly(hourly_df)
    #     run_weekend_weekday_daily(daily_df)

    # if args.run_artificial_removal or args.all:
    #     if os.path.exists(artificial_removal_chunks_path):
    #         os.remove(artificial_removal_chunks_path)
    #     if os.path.exists(artificial_removal_random_path):
    #         os.remove(artificial_removal_random_path)
    #     if os.path.exists(artificial_removal_sensitivity_analysis_path):
    #         os.remove(artificial_removal_sensitivity_analysis_path)
        
    #     test_artificial_removal_chunks(daily_df, run_for_season_daily)
    #     test_artificial_removal_chunks(hourly_df, run_for_season_hourly)

    #     test_artificial_removal_random(daily_df, run_for_season_daily)
    #     test_artificial_removal_random(hourly_df, run_for_season_hourly)
        
    #     perform_artificial_chunk_sensitivity_analysis(hourly_df, run_for_season_hourly)

    if args.run_times_of_day or args.all:
        if os.path.exists(time_of_day_hourly_path):
            os.remove(time_of_day_hourly_path)
        if os.path.exists(time_of_day_hourly_adjusted_path):
            os.remove(time_of_day_hourly_adjusted_path)
            
        for season in ["overall", "spring", "summer", "fall", "winter"]:
            run_times_of_day_hourly(hourly_df, season)
            run_times_of_day_hourly_adjusted(hourly_df, season)
