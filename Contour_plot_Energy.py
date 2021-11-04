# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:00:10 2021

@author: Mads Jorgensen
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = pypsa.Network("elec_s_37_lv2.0__Co2L0-solar+p3-dist10_2030.nc")

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

#%%

gen_tot = n.generators_t.p.sum(axis=1).sum()
gas = gen_tot.filter(like='gas')


#solar PV of the total capacity
solar_PV = n.generators_t.p.filter(like='solar').sum()
solar_PV = solar_PV[0:37].sum()
solar_PV_sum = solar_PV

#solar rooftop of the total capacity
solar_roof = n.generators_t.p.filter(like='solar roof').sum()
solar_roof_sum = solar_roof.sum()


capacity_roof = solar_roof_sum/gen_tot
capacity_PV = solar_PV_sum/gen_tot

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
co2_limits = 'Co2L0.05'
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
        gen_sum = n.generators_t.p.sum(axis=1).sum()
        
        
        onwind_calc = n.generators_t.p.filter(like='onwind').sum().sum()
        onwind[i,j] = onwind_calc/gen_sum*100
        
        offwind_ac_calc = n.generators_t.p.filter(like='offwind-ac').sum().sum()
        offwind_ac[i,j] = offwind_ac_calc/gen_sum*100
        
        offwind_dc_calc = n.generators_t.p.filter(like='offwind-dc').sum().sum()
        offwind_dc[i,j] = offwind_dc_calc/gen_sum*100
        
        
        solar_PV = n.generators_t.p.filter(like='solar').sum()
        solar_PV_calc = solar_PV[0:37].sum()
        solar_pv[i,j] = solar_PV_calc/gen_sum*100
        
        solar_roof_calc = n.generators_t.p.filter(like='solar roof').sum().sum()
        solar_roof[i,j] = solar_roof_calc/gen_sum*100
        
        gas_calc = n.generators_t.p.filter(like='gas').sum().sum()
        gas[i,j] = gas_calc/gen_sum*100
        
        
        j = j+1
    i = i +1        #adds one itteration to outer counter
    j = 0           #resets the inner counter
     
#%% Plotting data in contour plots


#plot for solar PV

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_pv, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of Produced energy - solar PV - Dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/05_solar_PV_dist_constant', dpi=300, bbox_inches='tight') 

#plot for solar roof

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_roof, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of Produced energy- solar rooftop - Dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/05_solar_roof_dist_constant', dpi=300, bbox_inches='tight') 
#plot for onwind

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(onwind, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of Produced energy - onwind - Dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/05_onwind_dist_constant', dpi=300, bbox_inches='tight') 

#plot for offwind ac

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_ac, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of Produced energy - offwind AC - Dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/05_offwind_ac_dist_constant', dpi=300, bbox_inches='tight') 

#plot for offwind dc

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_dc, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of Produced energy - offwind DC - Dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/05_offwind_dc_dist_constant', dpi=300, bbox_inches='tight') 

#plot for gas

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(gas, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Transmission line volume expansion")
ax.set_title("Contour plot of Produced energy - gas - Dist = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/05_gas_dist_constant', dpi=300, bbox_inches='tight') 

#%% Single loop for looping over dist + co2        

flex= 'elec_s_37'  
lv = '1.0'
co2_limits = 'Co2L0.05'
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
        gen_sum = n.generators_t.p.sum(axis=1).sum()
        
        
        onwind_calc = n.generators_t.p.filter(like='onwind').sum().sum()
        onwind[i,j] = onwind_calc/gen_sum*100
        
        offwind_ac_calc = n.generators_t.p.filter(like='offwind-ac').sum().sum()
        offwind_ac[i,j] = offwind_ac_calc/gen_sum*100
        
        offwind_dc_calc = n.generators_t.p.filter(like='offwind-dc').sum().sum()
        offwind_dc[i,j] = offwind_dc_calc/gen_sum*100
        
        
        solar_PV = n.generators_t.p.filter(like='solar').sum()
        solar_PV_calc = solar_PV[0:37].sum()
        solar_pv[i,j] = solar_PV_calc/gen_sum*100
        
        solar_roof_calc = n.generators_t.p.filter(like='solar roof').sum().sum()
        solar_roof[i,j] = solar_roof_calc/gen_sum*100
        
        gas_calc = n.generators_t.p.filter(like='gas').sum().sum()
        gas[i,j] = gas_calc/gen_sum*100
        
        
        j = j+1
    i = i +1        #adds one itteration to outer counter
    j = 0           #resets the inner counter
     
#%% Plotting data in contour plots


#plot for solar PV

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_pv, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - solar PV - lvl = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/06_solar_PV_lvl_constant', dpi=300, bbox_inches='tight') 

#plot for solar roof

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_roof, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy- solar rooftop - lvl = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/06_solar_roof_lvl_constant', dpi=300, bbox_inches='tight') 
#plot for onwind

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(onwind, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - onwind - lvl = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/06_onwind_lvl_constant', dpi=300, bbox_inches='tight') 

#plot for offwind ac

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_ac, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - offwind AC - lvl = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/06_offwind_ac_lvl_constant', dpi=300, bbox_inches='tight') 

#plot for offwind dc

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_dc, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - offwind DC - lvl = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/06_offwind_dc_lvl_constant', dpi=300, bbox_inches='tight') 

#plot for gas

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(gas, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("CO2 limit compared to 1990")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - gas - lvl = 1",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/06_gas_lvl_constant', dpi=300, bbox_inches='tight') 


#%% looping over distribution cost and transmission expansion

flex= 'elec_s_37'  
lv = '1.0'
co2_limits = 'Co2L0.05'
solar = 'solar+p3'
dist = '1'
dist = ['0.1','0.5','1','2','10']  # '1'
#co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '1.2', '1.5', '2.0']

i = 0       #start of the outer itterator
j = 0       #start of the inner itterator

for dis in dist:
    
    index1 = dist
    index2 = lvl
    
    for lv in lvl:
        
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limits+ '-' + solar +'-'+'dist'+dis+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        gen_sum = n.generators_t.p.sum(axis=1).sum()
        
        
        onwind_calc = n.generators_t.p.filter(like='onwind').sum().sum()
        onwind[i,j] = onwind_calc/gen_sum*100
        
        offwind_ac_calc = n.generators_t.p.filter(like='offwind-ac').sum().sum()
        offwind_ac[i,j] = offwind_ac_calc/gen_sum*100
        
        offwind_dc_calc = n.generators_t.p.filter(like='offwind-dc').sum().sum()
        offwind_dc[i,j] = offwind_dc_calc/gen_sum*100
        
        
        solar_PV = n.generators_t.p.filter(like='solar').sum()
        solar_PV_calc = solar_PV[0:37].sum()
        solar_pv[i,j] = solar_PV_calc/gen_sum*100
        
        solar_roof_calc = n.generators_t.p.filter(like='solar roof').sum().sum()
        solar_roof[i,j] = solar_roof_calc/gen_sum*100
        
        gas_calc = n.generators_t.p.filter(like='gas').sum().sum()
        gas[i,j] = gas_calc/gen_sum*100
        
        
        j = j+1
    i = i +1        #adds one itteration to outer counter
    j = 0           #resets the inner counter
     
#%% Plotting data in contour plots


#plot for solar PV

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_pv, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("Transmission line volume expansion")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - solar PV - CO2 = 0.05",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/07_solar_PV_CO2_cons', dpi=300, bbox_inches='tight') 

#plot for solar roof

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(solar_roof, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("Transmission line volume expansion")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy- solar rooftop - CO2 = 0.05",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/07_solar_roof_CO2_cons', dpi=300, bbox_inches='tight') 
#plot for onwind

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(onwind, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("Transmission line volume expansion")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - onwind - CO2 = 0.05",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/07_onwind_CO2_cons', dpi=300, bbox_inches='tight') 

#plot for offwind ac

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_ac, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("Transmission line volume expansion")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - offwind AC - CO2 = 0.05",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/07_offwind_ac_CO2_cons', dpi=300, bbox_inches='tight') 

#plot for offwind dc

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(offwind_dc, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("Transmission line volume expansion")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - offwind DC - CO2 = 0.05",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/07_offwind_dc_CO2_cons', dpi=300, bbox_inches='tight') 

#plot for gas

fig, ax = plt.subplots(figsize=(15,10))

im, cbar = heatmap(gas, index1, index2, ax=ax, clim = [0,100],
                   cmap="YlGn", cbarlabel="Produced energy [%]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
ax.set_xlabel("Transmission line volume expansion")
ax.set_ylabel("Investment cost of distribution grid")
ax.set_title("Contour plot of Produced energy - gas - CO2 = 0.05",fontsize=15)
fig.tight_layout()
plt.show()

fig.savefig('Plots/07_gas_CO2_cons', dpi=300, bbox_inches='tight') 
