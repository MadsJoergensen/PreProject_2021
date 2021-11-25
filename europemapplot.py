# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:44:22 2021

@author: Anders
"""
        
#%% European map plot
import pypsa, os
#pandas package is very useful to work with imported data, time series, matrices ...
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import cartopy.crs as ccrs

#pathtest = r'C:\Users\ander\OneDrive - Aarhus Universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files\postnetworks/'
#network_name = pathtest+'elec_s_37_lv1.0__Co2L0.1-solar+p3-dist0.5_2030.nc'
path = r'C:/Users/ander/OneDrive - Aarhus Universitet/Maskiningenioer/Kandidat/3. semester/PreProject Master/WP4/Network_files_1hr/'
pathplot = r'C:/Users/ander/OneDrive - Aarhus Universitet/Maskiningenioer/Kandidat/3. semester/PreProject Master/WP4/Plots/map/'
network_name = path+'elec_s_37_lv1.0__H-T-H-solar+p3-dist1_2030mid.nc'


n = pypsa.Network('elec_s_37_lv1.0__Co2L0.05-solar+p3-dist2_2030.nc')

#%%
#df=n.buses.loc["ES0 0 low voltage"].rename('ES0 0')
#df=df.to_frame().T
#n.buses=pd.concat([df, n.buses])

# This works:
# n.generators['bus']=n.generators['bus'].replace({'ES0 0 low voltage':'ES0 0'})

# This does not work :).
#df=n.generators.loc['bus']["ES0 0 low voltage"].rename('ES0 0')
#df=df.to_frame().T
#n.buses=pd.concat([df, n.buses])

# Change bus name for solar rooftop to the same as for the other generators.
countries=n.generators['bus']
for i in countries:
    n.generators['bus']=n.generators['bus'].replace({str(i)+' low voltage':str(i)})



#buselec = n.generators.assign(g = n.generators_t.p.mean()).groupby(['bus', 'carrier']).g.sum()
#buselec = n.generators.assign(g = n.generators.p_nom.sum()).groupby(['bus', 'carrier']).g.sum()
buselec = n.generators.assign(g = n.generators_t.p.sum()).groupby(['bus', 'carrier']).g.sum()
#buselec = n.generators_t.p_nom.groupby(['bus', 'carrier'])
#buselec=buselec.drop('residential rural solar thermal', level='carrier')
#buselec=buselec.drop('services rural solar thermal', level='carrier')
#buselec=buselec.drop('urban rural solar thermal', level='carrier')
#buselec=buselec.drop('urban central solar thermal', level='carrier')
#buselec=buselec.drop('services urban decentral solar thermal', level='carrier')
#buselec=buselec.drop('residential urban decentral solar thermal', level='carrier')
#buselec=buselec.drop('ror', level='carrier')
#buselec=buselec.drop('uranium', level='carrier')
#buselec=buselec.drop('oil', level='carrier')
#buselec=buselec.drop('gas', level='carrier')
#buselec=buselec.drop('lignite', level='carrier')
#buselec=buselec.drop('coal', level='carrier')
# buselec=buselec.drop('offwind-ac', level='carrier')
# buselec=buselec.drop('offwind-dc', level='carrier')
# buselec=buselec.drop('offwind', level='carrier')
# buselec=buselec.drop('solar', level='carrier')
# buselec=buselec.drop('solar rooftop', level='carrier')

busline = n.storage_units.assign(g = n.storage_units_t.p.sum()).groupby(['bus', 'carrier']).g.sum().filter(like="hydro")
buslink = -n.links.assign(g = n.links_t.p1.sum()).groupby(['bus1', 'carrier']).g.sum().filter(like="GT")
buselec = buselec.append(busline)
buselec = buselec.append(buslink)

#color={'onwind':'#235ebc','offwind':'#6895dd','offwind-ac':'#6895dd','offwind-dc':'#6895dd','solar':'#f9d002','solar rooftop':'#ffea80','gas':'r'}
color={'onwind':'#235ebc','offwind':'#6895dd','offwind-ac':'#6895dd','offwind-dc':'#6895dd','solar':'#f9d002','solar rooftop':'#ffea80'
       ,'ror':'#78AB46','hydro':'#3B5323','PHS':'g','gas':'brown','OCGT':'brown','CCGT':'brown','uranium':'r','oil':'#B5A642','coal':'k','residential rural solar thermal':'m',
       'services rural solar thermal':'m','urban rural solar thermal':'g','residential urban decentral solar thermal':'g','lignite':'g',
       'urban central solar thermal':'m','nuclear':'r','services urban decentral solar thermal':'m'}


from matplotlib.patches import Circle, Ellipse
import matplotlib.patches as mpatches



#n.buses.loc["ES0 0 low voltage"]
#n.buses.loc["ES0 0 low voltage",["x","y"]] = [-3.43431,40.6009]
#n.buses.loc["ES0 0 low voltage",['carrier']] = ['AC']

#n.buses.loc["ES0 0 low voltage",['Name']] = ['ES0 0']         
     
#Filtering of the links that are DC Links
n.links=n.links.loc[n.links['carrier'] == 'DC']
   
                
fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
n.plot(bus_sizes=buselec/8e7, 
       bus_colors=color,
       color_geomap=True,
       boundaries=([-12,30,36,65]),
       #boundaries=([-10.2, 29, 35,  64]),
       branch_components=["Link","Line"],# ["Link","Line"] this one decides if we want the links 
       ax=ax)

#ax.set_title('Produced energy by wind and solar')
gas_patch = mpatches.Patch(color='Brown', label=' Gas')
hydro_patch = mpatches.Patch(color='#3B5323', label=' Hydro')
ror_patch = mpatches.Patch(color='#78AB46', label=' Ror')
onwind_patch = mpatches.Patch(color='#235ebc', label=' Onwind')
offwind_patch = mpatches.Patch(color='#6895dd', label=' Offwind')
solar_patch = mpatches.Patch(color='#f9d002', label=' Solar Utility')
solarroof_patch = mpatches.Patch(color='#ffea80', label=' Solar Rooftop')
#ror_patch = mpatches.Patch(color='#78AB46', label=' Run-of-river')
ax.legend(handles=[onwind_patch,offwind_patch,solar_patch,solarroof_patch,gas_patch,hydro_patch,ror_patch],
        loc="lower right", bbox_to_anchor=(1.32, 0.5),#bbox_to_anchor=(0.01, 0.79),
        framealpha=0,   #color of the background of the legend
        handletextpad=0., columnspacing=0.5, ncol=1, title=None)


path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'
name = r'\02_map_dist2'
plt.savefig(path+name,dpi=300, bbox_inches='tight')


#%% Using the correct plot with transmission line expansion

loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom_opt*n.lines.s_max_pu).sort_index()).fillna(0.)

fig,ax = plt.subplots(subplot_kw = {"projection": ccrs.PlateCarree()})

n.plot(bus_sizes=buselec/8e7, 
       bus_colors=color,
       ax=ax,
       #bus_colors='gray',
       branch_components=["Line"],
       line_widths=n.lines.s_nom_opt/9e3,
       line_colors='purple',
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       boundaries=([-12,30,36,65]))
       #bus_sizes=0.1)
#ax.set_title('Produced energy by wind and solar')
gas_patch = mpatches.Patch(color='Brown', label=' Gas')
hydro_patch = mpatches.Patch(color='#3B5323', label=' Hydro')
ror_patch = mpatches.Patch(color='#78AB46', label=' Ror')
onwind_patch = mpatches.Patch(color='#235ebc', label=' Onwind')
offwind_patch = mpatches.Patch(color='#6895dd', label=' Offwind')
solar_patch = mpatches.Patch(color='#f9d002', label=' Solar Utility')
solarroof_patch = mpatches.Patch(color='#ffea80', label=' Solar Rooftop')
#ror_patch = mpatches.Patch(color='#78AB46', label=' Run-of-river')
ax.legend(handles=[onwind_patch,offwind_patch,solar_patch,solarroof_patch,gas_patch,hydro_patch,ror_patch],
        loc="lower right", bbox_to_anchor=(1.32, 0.5),#bbox_to_anchor=(0.01, 0.79),
        framealpha=0,   #color of the background of the legend
        handletextpad=0., columnspacing=0.5, ncol=1, title=None)

path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'
name = r'\02_map_dist2'
plt.savefig(path+name,dpi=300, bbox_inches='tight')

