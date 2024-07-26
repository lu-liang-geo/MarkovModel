import pandas
import giddy
import seaborn
import matplotlib.pyplot as plt
import matplotlib

import helper_utils
import classic_markov, spatial_markov
from markov_helpers import *

greens_adjusted = plt.get_cmap('Greens')(range(256))
#greens_adjusted[:1, :] = numpy.array([1, 1, 1, 1])
greens_adjusted = greens_adjusted[130:]
greens_dark_color = matplotlib.colors.ListedColormap(colors=greens_adjusted, name="greens_dark")
plt.register_cmap(cmap=greens_dark_color)

real_states = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
num_real_states = len(real_states)

daily_df = pandas.read_csv("data/Full PA Data/PA_geotagged_daily.csv")
daily_df = helper_utils.filter_time(daily_df)

classic_daily_result = classic_markov.run_for_season_daily(daily_df, "overall", save_to_seasonal=False)
spatial_matrices = spatial_markov.run_for_season_daily(daily_df, "overall", save_to_seasonal=False)

matrix_list = []
matrix_list.append(classic_daily_result.transition_count_matrix)
matrix_list.append(spatial_matrices[0])

print("\n------------------------\n")

print("Classic vs. Spatial Good Lag Hypothesis Test:")
classic_good_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
												regime_names=["Classic", "Spatial Good Lag"])
print(classic_good_results.summary())

matrix_list = []
matrix_list.append(classic_daily_result.transition_count_matrix)
matrix_list.append(spatial_matrices[1])
print("Classic vs. Spatial Moderate Lag Hypothesis Test:")
classic_moderate_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
												regime_names=["Classic", "Spatial Moderate Lag"])
print(classic_moderate_results.summary())

matrix_list = []
matrix_list.append(spatial_matrices[0])
matrix_list.append(spatial_matrices[1])
print("Spatial Good Lag vs. Spatial Moderate Lag Hypothesis Test:")
good_moderate_results = giddy.markov.homogeneity(matrix_list, class_names=real_states,
												regime_names=["Spatial Good Lag", "Spatial Moderate Lag"])
print(good_moderate_results.summary())

#Classic, Good, Moderate
plot_matrix = numpy.array([[0, classic_good_results.Q, classic_moderate_results.Q],
               [classic_good_results.Q, 0, good_moderate_results.Q],
               [classic_moderate_results.Q, good_moderate_results.Q, 0]])
pos1 = 0.8
pos2 = 2
pos3 = 3.3
x = [pos1 - .1, pos1 - .1, pos1 - .15, pos2, pos2, pos3 + .09]
y = [pos1 - .1, pos2, pos3 - .05, pos1 - .07, pos2, pos1 - .03]
values = numpy.array([classic_moderate_results.Q, classic_good_results.Q, 1,
                     good_moderate_results.Q, 1, 1])

plt.cla()
fig, ax = plt.subplots(dpi=400)
im = ax.scatter(x, y, s=values, c=values, cmap="greens_dark")

base_size = 90
values *= base_size
ax.scatter(x, y, s=values, c=values, cmap="greens_dark")

models = ["Classic", "Spatial\nGood Lag", "Spatial\nModerate Lag"]
models_reversed = models
models_reversed.reverse()

#numpy.arange(len(models)) + 1
ax.set_xticks([pos3, pos2, pos1], models, weight="bold", fontsize=12)
ax.set_yticks([0.8, 2, 3.3], models_reversed, weight="bold", fontsize=12)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

ax.vlines([4/3, 8/3, 4], 0, [4, 8/3, 4/3], color="black")
ax.hlines([4/3, 8/3, 4], 0, [4, 8/3, 4/3], color="black")

ax.spines[["top", "right"]].set_visible(False)

fig.colorbar(im)
fig.savefig("plots/ClassicSpatialDotHeatmap.png", bbox_inches="tight")

