import pandas
import giddy
import numpy
import matplotlib.pyplot as plt
import helper_utils
import os
import sys
import geopandas
import libpysal
from geopy import distance
import matplotlib
from markov_helpers import *


states = ["Missing", "Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
num_states = len(states)

real_states = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
num_real_states = len(real_states)

base_output_dir = os.path.join("results", "spatial_markov_raw_filtered")

seasonal_path = os.path.join(base_output_dir, "seasonal_6km.txt")

#Make custom colormaps

color_array = plt.get_cmap('binary')(range(256))
color_array[:,-1] = numpy.linspace(0.0, 0.5, 256)

map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name='greys_alpha', colors=color_array)
#plt.register_cmap(cmap=map_object)

reds_adjusted = plt.get_cmap('Reds')(range(256))
reds_adjusted[:1, :] = numpy.array([1, 1, 1, 1])
reds_white_color = matplotlib.colors.ListedColormap(colors=reds_adjusted, name="reds_white")
#plt.register_cmap(cmap=reds_white_color)


def generate_spatial_weight_matrix(location_ids):
    metadata_df = pandas.read_csv("data/PA_metadata.csv")
    metadata_df = metadata_df[metadata_df["Location 1 ID"].isin(location_ids)]
    gdf = geopandas.GeoDataFrame(metadata_df, geometry=geopandas.points_from_xy(
		metadata_df["Location 1 Longitude"], metadata_df["Location 1 Latitude"]
	))
    
    neighbor_distance_km = 6.036
    neighbors = {}
    weights = {}
    
    for base_index, base_row in gdf.iterrows():
        base_coords = (base_row["Location 1 Latitude"], base_row["Location 1 Longitude"])
        base_neighbors = []
        base_weights = []
        
        if base_row["Location 1 Latitude"] == "" or base_row["Location 1 Longitude"] == "":
                continue
        
        for neighbor_index, neighbor_row in gdf.iterrows():
            if base_index == neighbor_index:
                continue
            
            if neighbor_row["Location 1 Latitude"] == "" or neighbor_row["Location 1 Longitude"] == "":
                continue
            
            neighbor_coords = (neighbor_row["Location 1 Latitude"], neighbor_row["Location 1 Longitude"])
            
            try:
                distance_km = distance.distance(base_coords, neighbor_coords).km
            except:
                #print(base_coords)
                #print(neighbor_coords)
                continue

            if distance_km < neighbor_distance_km:
                base_neighbors.append(neighbor_row["Location 1 ID"])
                base_weights.append(1 / (neighbor_distance_km**2))
                
        neighbors[str(base_row["Location 1 ID"])] = base_neighbors
        weights[str(base_row["Location 1 ID"])] = base_weights
        
    weight_matrix = libpysal.weights.W(neighbors, weights=weights)
    weight_matrix.transform = 'R'
    return weight_matrix


def plot_locations(metadata_df):
    gdf = geopandas.GeoDataFrame(metadata_df, geometry=geopandas.points_from_xy(
		metadata_df["Location 1 Longitude"], metadata_df["Location 1 Latitude"]
	))
    
    county_data = geopandas.read_file("Tx_Census_CntyGeneralCoast_TTU/Tx_Census_CntyGeneralCoast_TTU.shp")
    ax = county_data[county_data["NAME"] == "Denton County"].plot(color='lightblue', edgecolor='black')
    gdf.plot(ax=ax)
    plt.show()


def run_for_season_daily(df, season="overall", path="plots/Spatial Markov Heatmaps/", save_to_seasonal=True):
    if save_to_seasonal:
        file = open(seasonal_path, "a")
        sys.stdout = file
    
    print("----" + season.capitalize() +", Daily----")
    
    season_df = helper_utils.filter_season(df, season)
    season_df = helper_utils.impute_missing_values_daily(season_df)
    location_ids = season_df["Location ID"].unique()

    #Is indexing of location ID's the same as indexing in the weight matrix?

    values = []
    for location_id in location_ids:
        location_rows = season_df[season_df["Location ID"] == location_id]
        location_pm25 = location_rows["PM2.5 (ATM)"].values.flatten().tolist()
        
        values.append(location_pm25)
    
    weights = generate_spatial_weight_matrix(location_ids)
    
    cutoffs = [-0.1, 12, 35.4, 55.4, 150.4, 250.4]
    spatial_markov_model = giddy.markov.Spatial_Markov(numpy.array(values), weights, 
                                                       fixed=True, k=7, m=7,
                                                       cutoffs=cutoffs, lag_cutoffs=cutoffs)
    
    numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    results_list = []
    matrix_list = []
    transition_count_list = []
    
    for i in range(1, len(spatial_markov_model.T)):
        print("Transition Matrix for Average Lag", states[i])
        print(str(states[1:]) + " | Sample Size")
        
        transition_counts = spatial_markov_model.T[i]
        new_transition_counts = []
        for row in transition_counts:
            transition_count_row = numpy.delete(row, 0)
            new_transition_counts.append(transition_count_row)
            
        del new_transition_counts[0]
        transition_prob_matrix = []
        transition_sample_sizes = []
        for transition_count_row in new_transition_counts:
            total_transitions = transition_count_row.sum()
            transition_probability_row = transition_count_row
            
            if total_transitions > 0:
                transition_probability_row = numpy.multiply(
                    transition_count_row, 1.0/total_transitions)
            
            transition_prob_matrix.append(transition_probability_row.tolist())
            transition_sample_sizes.append(total_transitions)
            
            print(str(transition_probability_row) + " | " + str(total_transitions))
            
        print()
        
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
        
        result = ClassicMarkovResult(
            numpy.array(new_transition_counts), transition_prob_matrix, transition_sample_sizes
        )
        transition_count_list.append(numpy.array(new_transition_counts))
        
        results_list.append(result)
        matrix_list.append(result.transition_count_matrix)
        adjusted_prob_matrix = remove_low_sample_size_states(result)
        
        if len(adjusted_prob_matrix) > 0:
            print("Adjusted Steady State: ")
            print(giddy.ergodic.steady_state(numpy.array(adjusted_prob_matrix)))
            print()

        print("Sojourn Time")
        print(giddy.markov.sojourn_time(transition_prob_matrix))
        
        print("\n\n")
        
        plt.cla()
        fig, ax = plt.subplots()

        threshold = 150 #TODO: Validate, possibly change for daily/hourly
        
        size_per_entry = 1
        plot_size = size_per_entry * num_real_states
        
        #row = [0 for _k in range(plot_size)]
        transition_matrix_to_plot = []
        #transition_matrix_to_plot = [row for _k in range(plot_size)]
        #print(transition_matrix_to_plot, file=sys.stderr)
        overlay_matrix = []
        
        for _i in range(plot_size):
            transition_matrix_to_plot.append([])
            overlay_matrix.append([])
        
        for j in range(num_real_states):
            for k in range(num_real_states):
                val = transition_prob_matrix[j][k]
                
                for x in range(size_per_entry):
                    for y in range(size_per_entry):
                        overlay_val = 0
                        regular_val = val
                        
                        if transition_sample_sizes[j] < threshold:
                            overlay_val = 1.0
                        
                        transition_matrix_to_plot[j*size_per_entry + x].append(regular_val)
                        overlay_matrix[j*size_per_entry + x].append(overlay_val)
        
        im = ax.imshow(transition_matrix_to_plot, cmap="reds_white", interpolation="none",
                    vmin=0.0, vmax=1.0)
        ax.imshow(overlay_matrix, cmap="greys_alpha" )
        
        fig.colorbar(im)
        fig.set_size_inches(8, 8)
        
        name = "Daily, Average Lag " + str(states[i])

        ax.set_title("Transition Probability Matrix for " + season.capitalize() + ", " + name)
        ax.set_xticks(numpy.arange(len(real_states)),
                    labels=real_states, rotation=45, ha="right")
        ax.set_yticks(numpy.arange(len(real_states)), labels=real_states)

        fig.savefig(os.path.join(path, season.capitalize(), name + ".png"))
    
    #Perform hypothesis tests
    print("Regular Hypothesis Test:")
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
                                                   regime_names=real_states)
    print(homogeneity_results.summary())
    
    print("Sample Size Adjusted Hypothesis Test:")
    matrix_list = generate_adjusted_matrix_list(results_list)
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
                                                   regime_names=real_states)
    print(homogeneity_results.summary())
    
    if save_to_seasonal:
        file.close()
    
    return transition_count_list


def run_for_season_hourly(df, season="overall", path="plots/Spatial Markov Heatmaps/"):
    file = open(seasonal_path, "a")
    sys.stdout = file
    
    print("----" + season.capitalize() +", Hourly----")
    
    season_df = helper_utils.filter_season(df, season)
    season_df = helper_utils.impute_missing_values_hourly(season_df)
    location_ids = season_df["Location ID"].unique()

    values = []
    for location_id in location_ids:
        location_rows = season_df[season_df["Location ID"] == location_id]
        location_pm25 = location_rows["PM2.5 (ATM)"].values.flatten().tolist()
        
        values.append(location_pm25)
    
    weights = generate_spatial_weight_matrix(location_ids)
    
    cutoffs = [-0.1, 12, 35.4, 55.4, 150.4, 250.4]
    spatial_markov_model = giddy.markov.Spatial_Markov(numpy.array(values), weights, 
                                                       fixed=True, k=7, m=7,
                                                       cutoffs=cutoffs, lag_cutoffs=cutoffs)
    
    numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    results_list = []
    matrix_list = []
    for i in range(1, len(spatial_markov_model.T)):
        print("Transition Matrix for Average Lag", states[i])
        print(str(states[1:]) + " | Sample Size")
        
        transition_counts = spatial_markov_model.T[i]
        new_transition_counts = []
        for row in transition_counts:
            transition_count_row = numpy.delete(row, 0)
            new_transition_counts.append(transition_count_row)
            
        del new_transition_counts[0]
        transition_prob_matrix = []
        transition_sample_sizes = []
        for transition_count_row in new_transition_counts:
            total_transitions = transition_count_row.sum()
            transition_probability_row = transition_count_row
            
            if total_transitions > 0:
                transition_probability_row = numpy.multiply(
                    transition_count_row, 1.0/total_transitions)
            
            transition_prob_matrix.append(transition_probability_row.tolist())
            transition_sample_sizes.append(total_transitions)
            
            print(str(transition_probability_row) + " | " + str(total_transitions))
            
        print()
        
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
        
        result = ClassicMarkovResult(
            numpy.array(new_transition_counts), transition_prob_matrix, transition_sample_sizes
        )
        results_list.append(result)
        matrix_list.append(result.transition_count_matrix)
        
        adjusted_prob_matrix = remove_low_sample_size_states(result)
        
        if len(adjusted_prob_matrix) > 0:
            print("Adjusted Steady State: ")
            print(giddy.ergodic.steady_state(numpy.array(adjusted_prob_matrix)))
            print()

        print("Sojourn Time")
        print(giddy.markov.sojourn_time(transition_prob_matrix))
        
        print("\n\n")
        
        plt.cla()
        fig, ax = plt.subplots()

        threshold = 150 #TODO: Validate, possibly change for daily/hourly
        
        size_per_entry = 1
        plot_size = size_per_entry * num_real_states
        
        #row = [0 for _k in range(plot_size)]
        transition_matrix_to_plot = []
        #transition_matrix_to_plot = [row for _k in range(plot_size)]
        #print(transition_matrix_to_plot, file=sys.stderr)
        overlay_matrix = []
        
        for _i in range(plot_size):
            transition_matrix_to_plot.append([])
            overlay_matrix.append([])
        
        for j in range(num_real_states):
            for k in range(num_real_states):
                val = transition_prob_matrix[j][k]
                
                for x in range(size_per_entry):
                    for y in range(size_per_entry):
                        overlay_val = 0
                        regular_val = val
                        
                        if transition_sample_sizes[j] < threshold:
                            overlay_val = 1.0
                        
                        transition_matrix_to_plot[j*size_per_entry + x].append(regular_val)
                        overlay_matrix[j*size_per_entry + x].append(overlay_val)
        
        im = ax.imshow(transition_matrix_to_plot, cmap="reds_white", interpolation="none",
                    vmin=0.0, vmax=1.0)
        ax.imshow(overlay_matrix, cmap="greys_alpha" )
        
        fig.colorbar(im)
        fig.set_size_inches(8, 8)
        
        name = "Hourly, Average Lag " + str(states[i])

        ax.set_title("Transition Probability Matrix for " + season.capitalize() + ", " + name)
        ax.set_xticks(numpy.arange(len(real_states)),
                    labels=real_states, rotation=45, ha="right")
        ax.set_yticks(numpy.arange(len(real_states)), labels=real_states)

        fig.savefig(os.path.join(path, season.capitalize(), name + ".png"))
        
    #Perform hypothesis tests
    print("Regular Hypothesis Test:")
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
                                                   regime_names=real_states)
    print(homogeneity_results.summary())
    
    print("Sample Size Adjusted Hypothesis Test:")
    matrix_list = generate_adjusted_matrix_list(results_list)
    homogeneity_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
                                                   regime_names=real_states)
    print(homogeneity_results.summary())
        
    file.close()



if __name__ == '__main__':
    daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
    daily_df = helper_utils.filter_time(daily_df)
    
    hourly_df = pandas.read_csv(
        "data/Full PA Data/PA_geotagged_hourly_raw_filtered.csv")
    hourly_df = helper_utils.filter_time(hourly_df)
    
    if os.path.exists(seasonal_path):
        os.remove(seasonal_path)
    
    #plot_locations(metadata_df)
    
    for season in ["overall", "spring", "summer", "fall", "winter"]:
        run_for_season_daily(daily_df, season)
        
    for season in ["overall", "spring", "summer", "fall", "winter"]:
        run_for_season_hourly(hourly_df, season)
    
    
