# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:59:24 2021

@author: Mads Jorgensen
"""
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

import matplotlib

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=90, va="top")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(color='black')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

#%% Looping over data - creating the arrays

onwind = np.zeros((5,5))
offwind_dc = np.zeros((5,5))
offwind_ac = np.zeros((5,5))
solar_pv = np.zeros((5,5))
solar_roof = np.zeros((5,5))
gas =  np.zeros((5,5))


#%% Single loop for looping over lvl + co2        

flex= 'elec_s_37'  
lv = '1.0'
co2_limits = 'Co2L0.5'
solar = 'solar+p3'
dist = '1'
#dist = ['0.1','0.5','1','2','10']  # '1'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '1.2', '1.5', '2.0']

i = 0       #start of the outer itterator
j = 0       #start of the inner itterator

for lv in lvl:
    
    index1 = lvl
    index2 = co2_limits
    
    for co2_limit in co2_limits:
        
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'-'+'dist'+dist+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        
        gen_sum = generators.sum()
        
        onwind[i,j] = generators.loc["onwind"]/gen_sum*100
        offwind_ac[i,j] = generators.loc["offwind-ac"]/gen_sum*100
        offwind_dc[i,j] = generators.loc["offwind-dc"]/gen_sum*100
        solar_pv[i,j] = generators.loc["solar"]/gen_sum*100
        solar_roof[i,j] = generators.loc["solar rooftop"]/gen_sum*100
        gas[i,j] = generators.loc["gas"]/gen_sum*100
        
        
        j = j+1
    i = i +1        #adds one itteration to outer counter
    j = 0           #resets the inner counter
     
#%% Plotting data in contour plots

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'


CO2_limit = ["0.5","0.2","0.1","0.05","0"]
#genreators = ["1","2","3","4","5","6","7"]

#plot for solar PV

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_pv, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - solar PV - dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/02_solar_PV', dpi=300, bbox_inches='tight') 

#plot for solar roof

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_roof, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - solar rooftop - dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/02_solar_roof', dpi=300, bbox_inches='tight') 
#plot for onwind

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(onwind, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - onwind - dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/02_onwind', dpi=300, bbox_inches='tight') 

#plot for offwind ac

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_ac, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - offwind AC - dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/02_offwind_ac', dpi=300, bbox_inches='tight') 

#plot for offwind dc

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_dc, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - offwind DC - dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/02_offwind_dc', dpi=300, bbox_inches='tight') 

#plot for gas

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(gas, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - gas - dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/02_gas', dpi=300, bbox_inches='tight') 

#%% Single loop for looping over co2 + dist       

flex= 'elec_s_37'  
lv = '1.0'
co2_limits = 'Co2L0.1'
solar = 'solar+p3'
#dist = '1'
dist = ['0.1','0.5','1','2','10']  # '1'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
#lvl = ['1.0', '1.1', '1.2', '1.5', '2.0']

i = 0       #start of the outer itterator
j = 0       #start of the inner itterator

for dis in dist:
    
    index1 = dist
    index2 = co2_limits
    
    for co2_limit in co2_limits:
        
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'-'+'dist'+dis+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        
        gen_sum = generators.sum()
        
        onwind[i,j] = generators.loc["onwind"]/gen_sum*100
        offwind_ac[i,j] = generators.loc["offwind-ac"]/gen_sum*100
        offwind_dc[i,j] = generators.loc["offwind-dc"]/gen_sum*100
        solar_pv[i,j] = generators.loc["solar"]/gen_sum*100
        solar_roof[i,j] = generators.loc["solar rooftop"]/gen_sum*100
        gas[i,j] = generators.loc["gas"]/gen_sum*100
        
        
        j = j+1
    i = i +1        #adds one itteration to outer counter
    j = 0           #resets the inner counter

#%% Plotting data in contour plots

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'


#plot for solar PV
fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_pv, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - solar PV - lvl = 1.0",fontsize=15)
fig.tight_layout()

fig.savefig('Plots/03_solar_PV', dpi=300, bbox_inches='tight') 


#plot for solar roof

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_roof, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - solar rooftop - lvl = 1.0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/03_solar_roof', dpi=300, bbox_inches='tight') 

#plot for onwind

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(onwind, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - onwind - lvl = 1.0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/03_onwind', dpi=300, bbox_inches='tight') 

#plot for offwind ac

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_ac, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - offwind AC - lvl = 1.0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/03_offwind_ac', dpi=300, bbox_inches='tight') 

#plot for offwind dc

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_dc, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - offwind DC - lvl = 1.0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/03_offwind_dc', dpi=300, bbox_inches='tight') 

#plot for gas

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(gas, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 level compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of installed capacity - gas - lvl = 1.0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/03_gas', dpi=300, bbox_inches='tight') 


#%% Single loop for looping over lvl + dist
        
flex= 'elec_s_37'  
lv = '1.0'
co2_limits = 'Co2L0.1'
solar = 'solar+p3'
dist = '1'
dist = ['0.1','0.5','1','2','10']  # '1'
#co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '1.2', '1.5', '2.0']

i = 0       #start of the outer itterator
j = 0       #start of the inner itterator

for lv in lvl:
    
    index1 = lvl
    index2 = dist
    
    for dis in dist:
        
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limits+ '-' + solar +'-'+'dist'+dis+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        
        gen_sum = generators.sum()
        
        onwind[i,j] = generators.loc["onwind"]/gen_sum*100
        offwind_ac[i,j] = generators.loc["offwind-ac"]/gen_sum*100
        offwind_dc[i,j] = generators.loc["offwind-dc"]/gen_sum*100
        solar_pv[i,j] = generators.loc["solar"]/gen_sum*100
        solar_roof[i,j] = generators.loc["solar rooftop"]/gen_sum*100
        gas[i,j] = generators.loc["gas"]/gen_sum*100
        
        
        j = j+1
    i = i +1        #adds one itteration to outer counter
    j = 0           #resets the inner counter   


#%% Plotting data in contour plots

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'


#plot for solar PV
fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_pv, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}%")
ax.set_ylabel("Transmission line volume expansion")
ax.set_xlabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - solar PV - CO2 = 0",fontsize=15)
fig.tight_layout()

fig.savefig('Plots/04_solar_PV', dpi=300, bbox_inches='tight') 


#plot for solar roof

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_roof, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}%")
ax.set_ylabel("Transmission line volume expansion")
ax.set_xlabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - solar rooftop - CO2 = 0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/04_solar_roof', dpi=300, bbox_inches='tight') 

#plot for onwind

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(onwind, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}%")
ax.set_ylabel("Transmission line volume expansion")
ax.set_xlabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - onwind - CO2 = 0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/04_onwind', dpi=300, bbox_inches='tight') 

#plot for offwind ac

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_ac, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}%")
ax.set_ylabel("Transmission line volume expansion")
ax.set_xlabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - offwind AC - CO2 = 0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/04_offwind_ac', dpi=300, bbox_inches='tight') 

#plot for offwind dc

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_dc, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}%")
ax.set_ylabel("Transmission line volume expansion")
ax.set_xlabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - offwind DC - CO2 = 0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/04_offwind_dc', dpi=300, bbox_inches='tight') 

#plot for gas

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(gas, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Installed capacity [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}%")
ax.set_ylabel("Transmission line volume expansion")
ax.set_xlabel("Investment cost of distribution grid")
ax.set_title("Contour plot of installed capacity - gas - CO2 = 0",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/04_gas', dpi=300, bbox_inches='tight') 


