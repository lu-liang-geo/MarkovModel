import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from PIL import Image


plt.rcParams["font.family"] = "Times New Roman"

color_array = plt.get_cmap('binary')(range(256))
color_array[:,-1] = numpy.linspace(0.0, 0.5, 256)

map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name='greys_alpha', colors=color_array)
plt.register_cmap(cmap=map_object)

reds_adjusted = plt.get_cmap('Reds')(range(256))
reds_adjusted[:1, :] = numpy.array([1, 1, 1, 1])
reds_white_color = matplotlib.colors.ListedColormap(colors=reds_adjusted, name="reds_white")
plt.register_cmap(cmap=reds_white_color)

abbreviated_states = ["G", "M", "USG", "U", "VU", "H"]
seasons = ["Overall", "Spring", "Summer", "Fall", "Winter"]


def plot_season_heatmaps_grid():
    fig, axes = plt.subplots(2, 5, dpi=400, width_ratios=[1, 1, 1, 1, 1])
    fig.tight_layout(h_pad=-16, w_pad=-2)
    
    overall_transition_matrix = [[0.8167433053827428, 0.17270760075737085, 0.010143359480659994, 0.0004057343792263998, 0.0, 0.0], [0.48138778460426457, 0.4994578966389592, 0.01915431875677629, 0.0, 0.0, 0.0], [0.15384615384615385, 0.8230769230769232, 0.023076923076923078, 0.0, 0.0, 0.0], [0.0, 0.75, 0.25, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    overall_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    im = axes[0, 0].imshow(overall_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[0, 0].imshow(overall_overlay_matrix, cmap="greys_alpha" )
    # colorbar = fig.colorbar(im, ax=axes[0, 0], orientation="horizontal")
    # tick_labels = colorbar.ax.get_xticklabels()
    # colorbar.ax.set_xticklabels(tick_labels, fontsize=8, weight="bold")

    spring_transition_matrix = [[0.8069135802469136, 0.1797530864197531, 0.013333333333333334, 0.0, 0.0, 0.0], [0.6090342679127726, 0.32398753894080995, 0.06697819314641744, 0.0, 0.0, 0.0], [0.013513513513513514, 0.9594594594594595, 0.02702702702702703, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    spring_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    axes[0, 1].imshow(spring_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[0, 1].imshow(spring_overlay_matrix, cmap="greys_alpha" )

    summer_transition_matrix = [[0.8961776859504132, 0.10382231404958678, 0.0, 0.0, 0.0, 0.0], [0.5360824742268041, 0.4639175257731959, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    summer_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    axes[0, 2].imshow(summer_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[0, 2].imshow(summer_overlay_matrix, cmap="greys_alpha" )

    fall_transition_matrix = [[0.7828162291169452, 0.2171837708830549, 0.0, 0.0, 0.0, 0.0], [0.38472622478386165, 0.6152737752161382, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    fall_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    axes[0, 3].imshow(fall_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[0, 3].imshow(fall_overlay_matrix, cmap="greys_alpha" )

    winter_transition_matrix = [[0.767984754645069, 0.20771796093377798, 0.022868032396379228, 0.0014292520247737017, 0.0, 0.0], [0.44790652385589097, 0.5423563777994158, 0.009737098344693282, 0.0, 0.0, 0.0], [0.33928571428571425, 0.6428571428571428, 0.017857142857142856, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    winter_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    
    axes[0, 4].imshow(winter_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[0, 4].imshow(winter_overlay_matrix, cmap="greys_alpha" )
    
    # overall_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Daily/Overall DailyNoTicksNoLegend.png")
    # spring_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Daily/Spring DailyNoTicksNoLegend.png")
    # summer_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Daily/Summer DailyNoTicksNoLegend.png")
    # fall_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Daily/Fall DailyNoTicksNoLegend.png")
    # winter_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Daily/Winter DailyNoTicksNoLegend.png")
    
    # for i, image in enumerate([overall_im, spring_im, summer_im, fall_im, winter_im]):
    #     axes[0, i].imshow(image)
    #     axes[0, i].axis("off")
        
        
    # overall_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Hourly/Overall Hourly.png")
    # spring_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Hourly/Spring HourlyNoTicksNoLegend.png")
    # summer_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Hourly/Summer HourlyNoTicksNoLegend.png")
    # fall_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Hourly/Fall HourlyNoTicksNoLegend.png")
    # winter_im = matplotlib.image.imread(
    #     "plots/Classic Markov Heatmaps/Seasonal Hourly/Winter HourlyNoTicksNoLegend.png")
    
    # for i, image in enumerate([overall_im, spring_im, summer_im, fall_im, winter_im]):
    #     axes[1, i].imshow(image)
    #     axes[1, i].axis("off")
    
    overall_transition_matrix = [[0.9622656051648472, 0.03726741866453883, 0.00035363243988248525, 0.00011334373073156578, 0.0, 0.0], [0.12826295140712368, 0.8510395106731334, 0.01972444471903865, 0.0009422013530629268, 3.0891847641407435e-05, 0.0], [0.0101010101010101, 0.23484848484848486, 0.7083333333333334, 0.04635642135642136, 0.00036075036075036075, 0.0], [0.009700889248181082, 0.0582053354890865, 0.20776071139854485, 0.7073565076798706, 0.01616814874696847, 0.0008084074373484236], [0.018867924528301886, 0.018867924528301886, 0.03773584905660377, 0.3584905660377358, 0.4528301886792453, 0.11320754716981132], [0.05263157894736842, 0.0, 0.0, 0.05263157894736842, 0.2631578947368421, 0.631578947368421]]
    overall_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    im = axes[1, 0].imshow(overall_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[1, 0].imshow(overall_overlay_matrix, cmap="greys_alpha" )
    
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes('bottom', size='5%', pad=-.5)
    cax.axis("off")
    
    position = axes[1, 0].get_position()
    y_offset = 0.045
    position.y0 = position.y0 + y_offset
    position.y1 = position.y1 + y_offset
    axes[1, 0].set_position(position)

    spring_transition_matrix = [[0.9620882036111066, 0.03725260060774636, 0.00048233837644902485, 0.00017685740469797578, 0.0, 0.0], [0.13774024461269657, 0.8387303436225976, 0.02253931275480489, 0.0009318578916715201, 5.8241118229470004e-05, 0.0], [0.011482254697286011, 0.20407098121085593, 0.7171189979123173, 0.06680584551148225, 0.0005219206680584551, 0.0], [0.006839945280437756, 0.03967168262653899, 0.1655266757865937, 0.7824897400820794, 0.004103967168262654, 0.0013679890560875513], [0.07692307692307693, 0.0, 0.07692307692307693, 0.23076923076923078, 0.3076923076923077, 0.3076923076923077], [0.0, 0.0, 0.0, 0.058823529411764705, 0.23529411764705882, 0.7058823529411764]]
    spring_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    axes[1, 1].imshow(spring_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[1, 1].imshow(spring_overlay_matrix, cmap="greys_alpha" )
    
    cbax = axes[1, 1].inset_axes([0.01, 0.15, 1, 1], transform=axes[1, 1].transAxes)
    cbax.axis("off")
    
    colorbar = plt.colorbar(im, ax=cbax, orientation="horizontal")
    tick_labels = colorbar.ax.get_xticklabels()
    colorbar.ax.set_xticklabels(tick_labels, fontsize=8, weight="bold")

    summer_transition_matrix = [[0.974194523261866, 0.025620792129077752, 8.394754957102802e-05, 0.00010073705948523363, 0.0, 0.0], [0.14478273164294467, 0.8520124422660006, 0.0029220473183146386, 0.0002827787727401263, 0.0, 0.0], [0.11290322580645161, 0.5161290322580645, 0.29032258064516125, 0.06451612903225806, 0.016129032258064516, 0.0], [0.26666666666666666, 0.06666666666666667, 0.4666666666666667, 0.13333333333333333, 0.06666666666666667, 0.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    summer_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    axes[1, 2].imshow(summer_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[1, 2].imshow(summer_overlay_matrix, cmap="greys_alpha" )

    fall_transition_matrix = [[0.9561181434599155, 0.043708116157855546, 0.0001737403822288409, 0.0, 0.0, 0.0], [0.11195672683816592, 0.8801182464305931, 0.007610541543493301, 0.0003144851877476571, 0.0, 0.0], [0.00975609756097561, 0.3097560975609756, 0.6731707317073171, 0.007317073170731707, 0.0, 0.0], [0.0, 0.375, 0.5, 0.125, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    fall_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    axes[1, 3].imshow(fall_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[1, 3].imshow(fall_overlay_matrix, cmap="greys_alpha" )

    winter_transition_matrix = [[0.9543602281988589, 0.044877927784274116, 0.0006201055951241982, 0.00014173842174267388, 0.0, 0.0], [0.12405757368060315, 0.8380005874865367, 0.036081464799765005, 0.001811416821697836, 4.895721139723881e-05, 0.0], [0.007292327203551046, 0.23779327837666456, 0.7162333544705136, 0.03868103994927077, 0.0, 0.0], [0.006211180124223603, 0.08074534161490683, 0.2587991718426501, 0.6211180124223603, 0.033126293995859216, 0.0], [0.0, 0.0, 0.02631578947368421, 0.42105263157894735, 0.5263157894736842, 0.02631578947368421], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    winter_overlay_matrix = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    
    axes[1, 4].imshow(winter_transition_matrix, cmap="reds_white", interpolation="none",
                   vmin=0.0, vmax=1.0)
    axes[1, 4].imshow(winter_overlay_matrix, cmap="greys_alpha" )
    
    for i in range(2):
         for j in range(5):
            if i == 1 and j == 0:
                continue
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)
    
    abbreviated_states = ["G", "M", "USG", "U", "VU", "H"]

    axes[1, 0].set_xticks(numpy.arange(len(abbreviated_states)),
                labels=abbreviated_states, rotation=45, fontsize=8, weight="bold")
    axes[1, 0].set_yticks(numpy.arange(len(abbreviated_states)), 
                labels=abbreviated_states, fontsize=8, weight="bold")
        
        
    fig.text(0.22, 0.72, "a")
    fig.text(0.4, 0.715, "b")
    fig.text(0.58, 0.72, "c")
    fig.text(0.76, 0.715, "d")
    fig.text(0.945, 0.72, "e")
    
    fig.text(0.22, 0.48, "f")
    fig.text(0.40, 0.49, "g")
    fig.text(0.58, 0.48, "h")
    fig.text(0.765, 0.48, "i")
    fig.text(0.95, 0.48, "j")
    
    fig.text(0.09, 0.76, "Whole Year", weight="bold")
    fig.text(0.3, 0.76, "Spring", weight="bold")
    fig.text(0.48, 0.76, "Summer", weight="bold")
    fig.text(0.675, 0.76, "Fall", weight="bold")
    fig.text(0.845, 0.76, "Winter", weight="bold")

    #fig.suptitle("Hourly Seasonal Transition Probability Heatmaps")
    plt.savefig("plots/Classic Markov Heatmaps/DailyHourlySeasonsGrid.png", 
                bbox_inches='tight')
    #plt.show()


def plot_filtered_heatmaps_grid(season):
    fig = plt.figure(figsize=(6, 6))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 3))

    Hourly_im = matplotlib.image.imread(
        "plots/Classic Markov Heatmaps/" + season.capitalize() +  " Hourly.png")
    threshold_25 = matplotlib.image.imread(
          "plots/Classic Markov Heatmaps/Filtered " + season.capitalize() +  " Hourly, 25% Threshold.png")
    threshold_50 = matplotlib.image.imread(
           "plots/Classic Markov Heatmaps/Filtered " + season.capitalize() +  " Hourly, 50% Threshold.png")

    for ax, im in zip(grid, [Hourly_im, threshold_25, threshold_50]):
        ax.axis("off")
        ax.imshow(im)

    fig.suptitle("Hourly Completeness Threshold Filtered Transition Probability Heatmaps")
    plt.savefig("plots/Classic Markov Heatmaps/Filtered/Filtered" + season.capitalize() +  "HeatmapsGrid.png")
    plt.show()


def plot_steady_state_stacked_chart():
    def plot_at_index(distributions, axes, fig, index):
        ax = axes[index]
        ax.xaxis.set_visible(False)
        ax.spines[["bottom", "top", "right"]].set_visible(False)
        
        if index > 0:
            ax.spines[["left"]].set_visible(False)

        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        left = numpy.zeros(5)
        for state, distribution in distributions.items():
            #y_positions = numpy.array([0, 1, 2, 3, 4]) * 0.08
            
            #height=0.045, 
            ax.barh(seasons, numpy.array(distribution), label=state, left=left,
                    align="center", height=0.6, edgecolor="black")
            
            # ax.set_ylim(-0.04, 0.36)
            # ax.set_yticks(y_positions)
            # ax.set_yticklabels(seasons)
            
            left += distribution
            
        if index != 0:
            ax.yaxis.set_visible(False)        
        if index == 4:
            ax.legend(loc="upper right", bbox_to_anchor=(2.1, 1.02), prop={'size': 8})

        for i in  range(len(distributions["Good"])):
            val = distributions["Good"][i]
            fig.text(0.16 + 0.155 * index, 0.17 + 0.153 * i, str(round(val * 100, 2)) + "%", weight="bold")
                      
    fig, axes = plt.subplots(1, 5, dpi=400)
    fig.set_size_inches(8, 1.9)
    fig.subplots_adjust(wspace=0.1)
    
    #Classic Markov daily
    good_distribution_by_season = [0.718, 0.74, 0.838, 0.639, 0.655]
    moderate_distribution_by_season = [0.269, 0.234, 0.162, 0.361, 0.325]
    usg_distribution_by_season = [0.013, 0.026, 0, 0, 0.018]
    unhealthy_distribution_by_season = [0, 0, 0, 0, 0.001]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 0)

    #Classic Markov hourly
    good_distribution_by_season = [0.756, 0.761, 0.849, 0.714, 0.699]
    moderate_distribution_by_season = [0.221, 0.207, 0.15, 0.279, 0.255]
    usg_distribution_by_season = [0.019, 0.023, 0.001, 0.007, 0.039]
    unhealthy_distribution_by_season = [0.004, 0.009, 0, 0, 0.006]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 1)
    
    #Spatial Markov daily with Good lag
    good_distribution_by_season = [0.772, 0.759, 0.851, 0.685, 0.682]
    moderate_distribution_by_season = [0.228, 0.221, 0.149, 0.315, 0.291]
    usg_distribution_by_season = [0, 0.02, 0, 0, 0.026]
    unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 2)
    
    #Spatial Markov hourly with Good lag
    good_distribution_by_season = [0.722, 0.746, 0.845, 0.734, 0.632]
    moderate_distribution_by_season = [0.262, 0.237, 0.154, 0.264, 0.321]
    usg_distribution_by_season = [0.017, 0.016, 0.001, 0.002, 0.042]
    unhealthy_distribution_by_season = [0, 0, 0, 0, 0.005]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 3)
    
    #Spatial Markov hourly with Moderate lag
    good_distribution_by_season = [0.676, 0.825, 0.789, 0.544, 0.640]
    moderate_distribution_by_season = [0.281, 0.154, 0.211, 0.456, 0.281]
    usg_distribution_by_season = [0.042, 0.020, 0, 0, 0.072]
    unhealthy_distribution_by_season = [0.001, 0, 0, 0, 0.007]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 4)
    
    #Label subplots
    fig.text(0.2, 0.05, "a", weight="bold")
    fig.text(0.35, 0.05, "b", weight="bold")
    fig.text(0.51, 0.05, "c", weight="bold")
    fig.text(0.665, 0.05, "d", weight="bold")
    fig.text(0.82, 0.05, "e", weight="bold")
    
    plt.savefig("plots/SteadyStates.png",  bbox_inches='tight')


def plot_steady_states_hourly():
    def plot_at_index(distributions, axes, fig, index):
        ax = axes[index]
        ax.xaxis.set_visible(False)
        ax.spines[["bottom", "top", "right"]].set_visible(False)
        
        if index > 0:
            ax.spines[["left"]].set_visible(False)

        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        left = numpy.zeros(5)
        for state, distribution in distributions.items():
            #y_positions = numpy.array([0, 1, 2, 3, 4]) * 0.08
            
            #height=0.045, 
            ax.barh(seasons, numpy.array(distribution), label=state, left=left,
                    align="center", height=0.6, edgecolor="black")
            
            # ax.set_ylim(-0.04, 0.36)
            # ax.set_yticks(y_positions)
            # ax.set_yticklabels(seasons)
            
            left += distribution
            
        if index != 0:
            ax.yaxis.set_visible(False)        
        if index == 4:
            ax.legend(loc="upper right", bbox_to_anchor=(2.1, 1.02), prop={'size': 8})

        for i in  range(len(distributions["Good"])):
            val = distributions["Good"][i]
            fig.text(0.18 + 0.27 * index, 0.17 + 0.153 * i, str(round(val * 100, 2)) + "%", weight="bold")
                      
    fig, axes = plt.subplots(1, 3, dpi=400)
    fig.set_size_inches(6, 1.9)
    fig.subplots_adjust(wspace=0.1)

    #Classic Markov hourly
    good_distribution_by_season = [0.756, 0.761, 0.849, 0.714, 0.699]
    moderate_distribution_by_season = [0.221, 0.207, 0.15, 0.279, 0.255]
    usg_distribution_by_season = [0.019, 0.023, 0.001, 0.007, 0.039]
    unhealthy_distribution_by_season = [0.004, 0.009, 0, 0, 0.006]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 0)
    
    #Spatial Markov hourly with Good lag
    good_distribution_by_season = [0.722, 0.746, 0.845, 0.734, 0.632]
    moderate_distribution_by_season = [0.262, 0.237, 0.154, 0.264, 0.321]
    usg_distribution_by_season = [0.017, 0.016, 0.001, 0.002, 0.042]
    unhealthy_distribution_by_season = [0, 0, 0, 0, 0.005]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 1)
    
    #Spatial Markov hourly with Moderate lag
    good_distribution_by_season = [0.676, 0.825, 0.789, 0.544, 0.640]
    moderate_distribution_by_season = [0.281, 0.154, 0.211, 0.456, 0.281]
    usg_distribution_by_season = [0.042, 0.020, 0, 0, 0.072]
    unhealthy_distribution_by_season = [0.001, 0, 0, 0, 0.007]
    very_unhealthy_distribution_by_season = [0, 0, 0, 0, 0]
    hazardous_distribution_by_season = [0, 0, 0, 0, 0]
    
    distributions = {
        "Good": good_distribution_by_season,
        "Moderate": moderate_distribution_by_season,
        "USG": usg_distribution_by_season,
        "Unhealthy": unhealthy_distribution_by_season,
        "Very Unhealthy": very_unhealthy_distribution_by_season,
        "Hazardous": hazardous_distribution_by_season
    }
    
    plot_at_index(distributions, axes, fig, 2)
    
    #Label subplots
    fig.text(0.18, 0.05, "Classic Markov", weight="bold")
    fig.text(0.44, -0.02, "Spatial Markov,\nGood Neighbors", weight="bold")
    fig.text(0.69, -0.02, "    Spatial Markov,\nModerate Neighbors", weight="bold")
    
    plt.savefig("plots/SteadyStatesHourly.png",  bbox_inches='tight')


def plot_sojourn_times():
    #Classic Markov hourly scale
    # state_sojourn_times = {
    #     "Good": (26.501, 26.377, 38.751, 22.788, 21.911),
    #     "Moderate": (6.713, 6.201, 6.757, 8.342, 6.173),
    #     "USG": (3.429, 3.535, 1.409, 3.060, 3.524),
    #     "Unhealthy": (3.417, 4.597, 1.154, 1.143, 2.639),
    #     "Very Unhealthy": (1.828, 1.444, 1, 0, 2.111),
    #     "Hazardous": (2.714, 3.4, 1, 0, 1)
    # }
    
    #Spatial Markov hourly scale, Good lag
    good_lag_sojourn_times = {
        "Good": (106.351, 104.078, 200.873, 127.533, 77.386),
        "Moderate": (39.159, 34.914, 36.752, 45.95, 41.03),
        "USG": (24.8, 24, 24, 24, 25.621),
        "Unhealthy": (24, 24, 24, 0, 24),
        "Very Unhealthy": (24, 24, 0, 0, 0),
        "Hazardous": (0, 24, 0, 0, 0)
    }
    
    #Spatial Markov hourly scale, Moderate lag
    moderate_lag_sojourn_times = {
        "Good": (90.574, 167.833, 184, 127.533, 80.889),
        "Moderate": (41.087, 36.969, 48.202, 45.94, 41.376),
        "USG": (25.66, 24.958, 24, 24, 26.789),
        "Unhealthy": (24, 0, 0, 0, 24),
        "Very Unhealthy": (0, 0, 0, 0, 0),
        "Hazardous": (24, 0, 0, 0, 0)
    }
        
    x = numpy.arange(len(seasons))
    width = 0.1
    
    fig, axes = plt.subplots(2, 1, layout='constrained', dpi=400, figsize=(6, 5))
    
    sojourn_times_by_lag = [good_lag_sojourn_times, moderate_lag_sojourn_times]
    for i in range(len(sojourn_times_by_lag)):
        state_sojourn_times = sojourn_times_by_lag[i]
        multiplier = 0
        for state, sojourn_times in state_sojourn_times.items():
            offset = width * multiplier
            rects = axes[i].bar(x + offset, sojourn_times, width, label=state, edgecolor="black")
            #ax.bar_label(rects, padding=3)
            multiplier += 1
            
        
        axes[i].grid(axis="y")
        axes[i].set_axisbelow(True)
        axes[i].set_ylim(0, 210)
        axes[i].tick_params(axis='y', which='major', labelsize=12)
        
    axes[1].tick_params(bottom = False)
    axes[1].set_xticks(x + width + 0.15, seasons, weight="bold", fontsize=14)
    axes[1].set_ylabel('Sojourn Time (Hours)', weight="bold", fontsize=12)
    axes[0].legend(loc='upper right', ncols=2, fontsize=10.5)
    axes[0].set_xticks([])
    
    fig.text(0.11, 0.92, "a", weight="bold", fontsize=18)
    fig.text(0.11, 0.43, "b", weight="bold", fontsize=18)
    
    fig.savefig("plots/Sojourn Times Spatial Markov.png")


def plot_state_diagram_grid():
    fig, axes = plt.subplots(1, 5, dpi=400, sharex=True, sharey=True)
    fig.set_size_inches(12, 5)
    fig.tight_layout(h_pad = -1.5, w_pad=-1.2)
    
    base_dir = "visualization_graphs"
    scale = "daily"
    
    for j in range(len(seasons)):
        ax = axes[j]
        
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #ax.spines[["bottom", "top", "left", "right"]].set_visible(False)
        
        season = seasons[j]
        
        file_name = f'{season.lower()}_{scale}.png'
        file_dir = os.path.join(base_dir, file_name)
        
        #img = matplotlib.image.imread(file_dir)
        img = Image.open(file_dir)
        
        desired_width = 1200
        desired_height = 1600
        
        width, height = img.size
        left = int((desired_width - width)/2)
        top = int((desired_height - height)/2) 
        
        new_img = Image.new(img.mode, (desired_width, desired_height), (255, 255, 255))
        new_img.paste(img, (left, top))
        
        ax.imshow(new_img)
            
    fig.savefig("plots/StateDiagramsDaily.png", transparent=True)


def combine_state_diagrams():
    fig, axes = plt.subplots(2, 1, dpi=400, sharex=True, sharey=True)
    fig.tight_layout(h_pad = -11)
    
    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis("off")
    
    daily_img = matplotlib.image.imread(os.path.join("plots", "StateDiagramsDaily.png"))
    hourly_img = matplotlib.image.imread(os.path.join("plots", "StateDiagramsHourly.png"))
    
    axes[0].imshow(daily_img)
    axes[1].imshow(hourly_img)
    
    fig.text(0.11, 0.86, "Whole Year", weight="bold", fontsize=12)
    fig.text(0.3, 0.86, "Spring", weight="bold", fontsize=12)
    fig.text(0.46, 0.86, "Summer", weight="bold", fontsize=12)
    fig.text(0.67, 0.86, "Fall", weight="bold", fontsize=12)
    fig.text(0.83, 0.86, "Winter", weight="bold", fontsize=12)
    
    #Manually create legend
    line1 = lines.Line2D([], [], color="white", marker='o', markersize=min(4, 0.5 * 1) * 8, markerfacecolor="lightgrey")
    line2 = lines.Line2D([], [], color="white", marker='o', markersize=0.5 * 3 * 8, markerfacecolor="lightgrey")
    line3 = lines.Line2D([], [], color="white", marker='o', markersize=0.5 * 5 * 8, markerfacecolor="lightgrey")
    fig.legend((line1, line2, line3), 
               ('1 Day', '3 Days', '5 Days'), 
               numpoints=1, loc=(0.12, 0.03))
    
    fig.savefig("plots/StateDiagrams.png")


if __name__ == '__main__':
    #plot_steady_state_stacked_chart()
    #plot_season_heatmaps_grid()
    #plot_sojourn_times()
    
    #plot_steady_states_hourly()
    
    #plot_state_diagram_grid()
    combine_state_diagrams()
    
    #for season in ["overall", "spring", "summer", "fall", "winter"]:
    #	plot_filtered_heatmaps_grid(season)