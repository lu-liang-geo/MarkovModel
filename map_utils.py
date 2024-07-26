import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def generate_axis():
    plt.cla()
    plt.axis("off")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    #ax.stock_img()
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.add_feature(cfeature.BORDERS)
        
    LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'],
                                            linewidth=.1)
    OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['water'],
                                            linewidth=.1)
    ax.add_feature(LAND, zorder=-1)
    ax.add_feature(OCEAN, zorder=0)
    
    return ax

def add_arrow_to_map(map):
    map.text(x=-97.1-0.37, y=32.9, s ='N', fontsize=20)
    map.arrow(-97.3, 33, 0, 0.18, length_includes_head=True,
          head_width=0.05, head_length=0.1, overhang=.2, facecolor='k')
    

def make_map_pretty(map):
    add_arrow_to_map(map)
    plt.xlim([-97.5, -96.8])
    plt.ylim([32.9, 33.5])
    
    #map.text(x=-96.5, y=27, s="Gulf of Mexico", fontsize=8)    
