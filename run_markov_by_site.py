import pandas
import os
from classic_markov import run_for_season_hourly
import helper_utils
import giddy
import numpy


if __name__ == "__main__":
    hourly_df = pandas.read_csv(
        "data/Full PA Data/PA_geotagged_hourly_raw_filtered.csv")
    hourly_df = helper_utils.filter_time(hourly_df)

    for season in ["overall", "spring", "summer", "fall", "winter"]:
        output = []
        locations = hourly_df["Location ID"].unique()
        for location in locations:
            location_df = hourly_df.query("`Location ID` == @location")
            if len(location_df) == 0:
                continue
            
            result = run_for_season_hourly(
				location_df, season=season, save_to_seasonal=False)
            
            prob_matrix = result.transition_prob_matrix
            steady_state = giddy.ergodic.steady_state(
				prob_matrix, fill_empty_classes=True)
            sojourn_time = giddy.markov.sojourn_time(prob_matrix)
            
            if len(prob_matrix) < 2:
                continue
            
            if numpy.ndim(steady_state) > 1:
                steady_state = steady_state[0]
                
            if sojourn_time[0] == numpy.inf or sojourn_time[1] == numpy.inf:
                continue
            
            output.append([location, steady_state[0], steady_state[1],
							sojourn_time[0], sojourn_time[1], len(location_df)])
            
        output_df = pandas.DataFrame(output, columns=["Location ID", "SteadyStateGood", "SteadyStateModerate",
														"SojournTimeGood", "SojournTimeModerate", "NumRecords"])
        output_df = output_df.query("NumRecords > 0")
        output_df.to_csv(f"results/markov_by_site/{season}.csv", index=False)
