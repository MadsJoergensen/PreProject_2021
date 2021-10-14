#!/usr/bin/env python
# coding: utf-8

# # Aarhus University - Fall 2021 - PreProject for Master Thesis 
# 
# This notebook includes the steps to optimize the capacity and dispatch of generators in the power system of one country.
# Make sure that you understand every step in this notebook. For the project of the course Renewable Energy Systems (RES) you need to deliver a report including the sections described at the end of this notebook.

# In[1]:


import pypsa
#pandas package is very useful to work with imported data, time series, matrices ...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

# We start by creating the network. In this example, the country is modelled as a single node, so the network will only include one bus.
# 
# We select the year 2015 and set the hours in that year as snapshots.
# 
# We select a country, in this case, Spain (ESP), and add one node (electricity bus) to the network.

n = pypsa.Network("elec_s_37_lv1.0__Co2L0.1-solar+p3-dist1_2030.nc")

# In[2]: counting the number of entries in the list of components

for c in n.iterate_components(list(n.components.keys())[2:]):
    print("Component '{}' has {} entries".format(c.name,len(c.df)))

print('\n','Temporal solution is = ',len(n.snapshots))


#%% Getting all the static data in the .nc file

generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
stores = n.stores.groupby("carrier")["e_nom_opt"].sum()
storage_unit = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
links = n.links.groupby("carrier")["p_nom_opt"].sum()

#%% print the different variables if you want

print(generators)       #printing the generators 
print(stores)           #printing the stores in the system
print(storage_unit)     #printing the storage units in the system
print(links)            #printing the links in the system

#%% Outputting the global constrains for CO2 price and total system cost

print(n.global_constraints.constant,'\n') #CO2 limit (constant in the constraint)
print(n.global_constraints.mu) #CO2 price (Lagrance multiplier in the constraint)

# In[7] Total Annual System Cost in billion euroes (devided by 1e9)

print('total system cost \n')
print(n.objective / 1e9)

#%% Plotting data in a bar and pie plot

label = generators.index            #getting the labels from a df 
col_map = plt.get_cmap('tab10')     #making a list of colors for plotting

def barplot(x,title,rotation):
    plt.figure(figsize=(15,5))
    plt.bar(x.index,x,color=col_map.colors)
    plt.title(title,fontsize=25)
    plt.xlabel('Generators',fontsize=15,color='r')
    plt.ylabel('Installed capacity \n [MW]',fontsize=15)
    plt.xticks(rotation=rotation)
    return

barplot(generators,'Bar plot of generators',90)     #plotting a barplot of variable

def pieplot(x,color): #choose from colors 'copper', 'hsv', 'jet', 'bwr'
    
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    fig1.subplots_adjust(0.3, 0, 1, 1)

    sizes = x
    labels = x.index    

    theme = plt.get_cmap(color)
    ax1.set_prop_cycle("color", [theme(1. * i / len(sizes))
                             for i in range(len(sizes))])
 
    _, _ = ax1.pie(sizes, startangle=90, radius=1800)
 
    ax1.axis('equal')
 
    total = sum(sizes)
    plt.legend(
        loc='upper left',
        labels=['%s, %1.1f%%' % (
            l, (float(s) / total) * 100)
                for l, s in zip(labels, sizes)],
        prop={'size': 11},
        bbox_to_anchor=(0.0, 1),
        bbox_transform=fig1.transFigure
        )
    plt.show()
    return

pieplot(generators,'jet')   #plotting the pie-plot from different data


#%% Plotting the network on the map

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'

help(n.plot)

loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom_opt*n.lines.s_max_pu).sort_index()).fillna(0.)

fig,ax = plt.subplots(subplot_kw = {"projection": ccrs.PlateCarree()})

n.plot(ax=ax,
       bus_colors='gray',
       branch_components=["Line"],
       line_widths=n.lines.s_nom_opt/9e3,
       line_colors=loading,
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       boundaries=([-12,30,36,65]),
       bus_sizes=0.1)
ax.axis('off');

# Save the figure in the selected path
name = r'\05_37' # Assign the name for the figure
plt.savefig(path+name, dpi=300, bbox_inches='tight')   

#%% Creating duration curves for technologies

generators = n.generators.groupby("carrier")["p_nom_opt"].sum()

df = n.generators

DK_onwind = df.loc["DK0 0 onwind", ["p_nom_opt"]]
DK_solar = df.loc["DK0 0 solar", ["p_nom_opt"]]
EU_gas = df.loc["EU gas", ["p_nom_opt"]]
EU_oil = df.loc["EU oil", ["p_nom_opt"]]


DK_t = n.generators_t.p.loc["2013","DK0 0 onwind"]
DK_s = n.generators_t.p.loc["2013","DK0 0 solar"]
EU_g = n.generators_t.p.loc["2013","EU gas"]
EU_o = n.generators_t.p.loc["2013","EU oil"]

#print(max(DK_t))
#print(DK_t.sort_values(axis=0, ascending=False))

x1 = DK_t.div(DK_onwind.sum()).sort_values(axis=0, ascending=False).to_numpy()
x2 = DK_s.div(DK_solar.sum()).sort_values(axis=0, ascending=False).to_numpy()
x3 = EU_g.div(EU_gas.sum()).sort_values(axis=0, ascending=False).to_numpy()
x4 = EU_o.div(EU_oil.sum()).sort_values(axis=0, ascending=False).to_numpy()

plt.plot(x1,label='onwind')
plt.plot(x2,label='solar')
plt.plot(x3,label='gas')
plt.plot(x4,label='oil')
plt.legend()
plt.title('Duration curve')
plt.xlabel('Hours',color='b')
plt.ylabel('Normalized Capacity',color='orange')
plt.axis([0,len(DK_t),0,1])
plt.show()

#plotting the gas consumption
plt.plot(EU_g,label='EU gas')
plt.plot(DK_t,label='DK onwind')
plt.legend()
#plt.axis(['2013-01','2014-01',0,max(DK_t)])
plt.show()
n.generators_t.p.filter(like='solar')#.sum(axis=1)

#%% Contour plots for installed capacity


import matplotlib.pyplot as plt

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()

plt.colorbar()
plt.show()


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
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
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
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



fig, ax = plt.subplots()

im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
plt.show()


# In[5] Static component data

print(n.lines.head())

print(n.generators.head())

print(n.storage_units.head())

# In[6] Time-varying component data (input data)

# plotting the load from all the countries in the network file

n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))


n.generators_t.p.sum(axis=1).plot(figsize=(15,3),color='r')



#plotting the solar generation for Italy in july
n.generators_t.p.loc["2013-07","IT0 0 solar"].plot(figsize=(15,3))

#plotting the onshore wind generation for Estonia in july
n.generators_t.p.loc["2013-07","DK0 0 onwind"].plot(figsize=(15,3))


# In[7] Total Annual System Cost

# Total system cost, in billion euroes

print(n.objective / 1e9)

# In[8] Transmission line Expansion

(n.lines.s_nom_opt - n.lines.s_nom).head(5)


# In[9] Optimal Generator/Storage Capacities

# Optimal generator units
n.generators.groupby("carrier").p_nom_opt.sum() /1e3

# Optimal storage units
n.storage_units.groupby("carrier").p_nom_opt.sum() /1e3


# In[10] Energy storage over time

(n.storage_units_t.state_of_charge.sum(axis=1).resample('D').mean() / 1e6).plot(figsize=(15,3))


(n.storage_units_t.state_of_charge["2013-07"].filter(like="PHS",axis=1)).plot(figsize=(15,3))

# In[11] plotting energy networks on maps

import cartopy.crs as ccrs

loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom_opt*n.lines.s_max_pu).sort_index()).fillna(0.)

# PlateCarree, Mercator, Orthographic

fig,ax = plt.subplots(
                subplot_kw = {"projection": ccrs.PlateCarree()}
    )

n.plot(ax=ax,
       bus_colors='gray',
       branch_components=["Line"],
       line_widths=n.lines.s_nom_opt/9e3,
       line_colors=loading,
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       bus_sizes=0.3)
ax.axis('off');

# Save the figure in the selected path
name = r'\01_map' # Assign the name for the figure
plt.savefig(path+name, dpi=300, bbox_inches='tight')   


#%% Building a stacked plot using pandas

flex= 'elec_s_37'  
lv = 'lv2.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-dist'

co2_limits=['0'] # the corresponding CO2 limits in the code


df = pd.DataFrame()


for co2_limit in co2_limits:      
        network_name= (flex+ '_' + lv + '__' +'Co2L'+co2_limit+ '-T-H-B-I-' + solar +'1'+'_'+'2030'+'.nc')        
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        
        df[co2_limit] = generators
        print(generators)

df["technology"] = df.index


#%%  Making a double for-loop

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'

flex= 'elec_s_37'  
lv = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3'
dist = '10'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '1.2', '1.5', '2.0'] #, '1.2', '1.5', '2.0'


df_g = pd.DataFrame()       #DataFrame for generator capacity
df_s = pd.DataFrame()       #DataFrame for storage capacity

for lv in lvl:
    for co2_limit in co2_limits:
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'-'+'dist'+dist+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        df_g[lv+co2_limit] = generators
        df_s[lv+co2_limit] = storage



#Filtering values for the generators, with the transmission expansion
df1 = df_g.filter(like='1.0')
df2 = df_g.filter(like='1.2')
df3 = df_g.filter(like='2.0')

plt.figure(1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,8),dpi = 200,sharey=True)
fig.suptitle('Installed capacity vs. CO2 constrain and Transmission expansion - Dist='+str(dist))
df1.plot.bar(ax=ax1,rot=25 )
ax1.set_xlabel('Carrier')
ax1.set_ylabel('Installed capacity [MW]')
ax1.set_title('Carrier capacity vs. CO2 emmisions - lv1.0')
ax1.set_ylim(0,700e3)
ax1.yaxis.grid()
ax1.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
#plt.rc('grid', linestyle="--", color='gray')
#ax1.legend(frameon = True, ncol = 5, shadow=True, bbox_to_anchor=(0.5, 1.25), loc='upper center', title = '% CO2 emmesion compared to 1990')

df2.plot.bar(ax=ax2,rot=25 )
ax2.set_xlabel('Carrier')
ax2.set_ylabel('Installed capacity [MW]')
ax2.set_title('Carrier capacity vs. CO2 emmisions - lv1.1')
ax2.set_ylim(0,700e3)
ax2.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax2.yaxis.grid()

df3.plot.bar(ax=ax3,rot=25 )
ax3.set_xlabel('Carrier')
ax3.set_ylabel('Installed capacity [MW]')
ax3.set_title('Carrier capacity vs. CO2 emmisions - lv2.0')
ax3.set_ylim(0,700e3)
ax3.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax3.yaxis.grid()

str(dist).replace('.', '')

# Save the figure in the selected path
name = r'\01_generator_dist='+str(dist).replace('.', '')
#plt.savefig(r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'+name, dpi=300,  bbox_inches='tight')
plt.savefig(path+name, dpi=300, bbox_inches='tight') 

# In[2]

#Filtering values for the storage capacity, with the transmission expansion

df4 = df_s.filter(like='1.0')
df5 = df_s.filter(like='1.1')
df6 = df_s.filter(like='2.0')

plt.figure(2)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,8),dpi = 200,sharey=True)
fig.suptitle('Installed capacity vs. CO2 constrain and Transmission expansion - Dist='+str(dist))
df4.plot.bar(ax=ax1,rot=25 )
ax1.set_xlabel('Carrier')
ax1.set_ylabel('Installed capacity [MW]')
ax1.set_title('Carrier capacity vs. CO2 emmisions - lv1.0')
ax1.set_ylim(0,120e3)
ax1.yaxis.grid()
ax1.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
#plt.rc('grid', linestyle="--", color='gray')
#ax1.legend(frameon = True, ncol = 5, shadow=True, bbox_to_anchor=(0.5, 1.25), loc='upper center', title = '% CO2 emmesion compared to 1990')

df5.plot.bar(ax=ax2,rot=25 )
ax2.set_xlabel('Carrier')
ax2.set_ylabel('Installed capacity [MW]')
ax2.set_title('Carrier capacity vs. CO2 emmisions - lv1.1')
ax2.set_ylim(0,120e3)
ax2.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax2.yaxis.grid()

df6.plot.bar(ax=ax3,rot=25 )
ax3.set_xlabel('Carrier')
ax3.set_ylabel('Installed capacity [MW]')
ax3.set_title('Carrier capacity vs. CO2 emmisions - lv2.0')
ax3.set_ylim(0,120e3)
ax3.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax3.yaxis.grid()

# Save the figure in the selected path
name = r'\01_storage_dist='+str(dist).replace('.', '') # Assign the name for the figure
plt.savefig(path+name, dpi=300, bbox_inches='tight')   

# In[3] Looking at the total system cost 

flex= 'elec_s_37'  
lv = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3'
dist = '10'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '1.2', '1.5', '2.0'] #, '1.2', '1.5', '2.0'

df_CO2 = pd.DataFrame()
df_cost = pd.DataFrame()

a = np.zeros((len(lvl),len(co2_limits)))

j = 0 
for lv in lvl:
    i = 0
    for co2_limit in co2_limits:
        
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'-'+'dist'+dist+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        CO2_price = n.global_constraints.constant
        System_cost = n.objective / 1e9 # Total system cost, in billion euroes
            
        df_CO2[lv+co2_limit] = CO2_price
        df_cost[lv+co2_limit] = n.global_constraints.mu
        a[j,i] = System_cost
        i = i+1
        
    j = j+1
        
plt.figure(figsize=(9,5))
x = ['50%','20%','10%','5%', '0%']
for i in range(len(a)):
    y = a[i,0:]
    
    #plt.figure(i)
    plt.plot(x,y,label = 'lv.'+str(lvl [i]))
    plt.title('Total system cost vs. transmission expansion- Dist='+str(dist),size = 18)
    plt.xlabel('CO2 - level compared to 1990',size = 15)
    plt.ylabel('System cost [Billion Euro]',size=15)
    plt.legend(prop={"size":12})

# Save the figure in the selected path
name = r'\01_system_cost_dist='+str(dist).replace('.', '') # Assign the name for the figure
plt.savefig(path+name, dpi=300, bbox_inches='tight')   
