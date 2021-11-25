# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:25:34 2021

@author: Mads Jorgensen
"""

import pypsa
#pandas package is very useful to work with imported data, time series, matrices ...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# We start by creating the network. In this example, the country is modelled as a single node, so the network will only include one bus.
# 
# We select the year 2015 and set the hours in that year as snapshots.
# 
# We select a country, in this case, Spain (ESP), and add one node (electricity bus) to the network.

n = pypsa.Network("elec_s_37_lv1.0__Co2L0.1-solar+p3-dist1_2030.nc")

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'

#%% Getting all the static data in the .nc file

generators = n.generators["p_nom_opt"].filter(like='solar roof')
stores = n.stores.groupby("carrier")["e_nom_opt"].sum()
storage_unit = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
links = n.links.groupby("carrier")["p_nom_opt"].sum()

generators.idxmin()
print('Looking at the maximum and minimum installed capacity of solar rooftop')
print('maximum value is at:', generators.idxmax())
print('minimum value is at:',generators.idxmin())

#%% extracting data from the network file

#test = n.buses_t.p
#test_df1 = test.filter(like='low voltage').sum(1)

#analysing the flow from High voltage to low voltage
links_t_p0 = n.links_t.p0.filter(like='distribution')

#looking at the flow over p1, which is from bus 0 to bus 1
links_t_p1 = n.links_t.p1.filter(like='distribution')

#check for the two different busses p0 and p1
test = (links_t_p0+links_t_p1).sum()

#extracting the data for the first different countries
links_t_NO = links_t_p0.filter(like='NO3')
links_t_IT = links_t_p0.filter(like='IT0')
links_t_AL = links_t_p0.filter(like='AL0')
links_t_ES = links_t_p0.filter(like='ES0')


#Finding the minimum value for the distribution grid
print(min(links_t_p0))

#%%

fig, ax = plt.subplots()

#Averaging factor (daily, weekly, monthly)
mu = 'D'

#averageing over each week for Norway
df1 = links_t_NO.resample(mu).mean()
df1.plot(ax=ax)
#averageing over each week for Italy
df2 = links_t_IT.resample(mu).mean()
df2.plot(ax=ax)

#averageing over each week for Spain
# df3 = links_t_ES.resample(mu).mean()
#df3.plot(ax=ax)
#averageing over each week for Albania

df4 = links_t_AL.resample(mu).mean()
df4.plot(ax=ax)

ax.legend(['Distribution grid','Load','Solar Rooftop'])
#ax.title(['Flow over distribution grid'])
ax.set_ylabel('Flow [MW]')
ax.set_xlabel('')
ax.set_title('mojn')
#%% Looking at the generation, distribution and load

fig, ax = plt.subplots()        #defining the plot

#averageing over each week for Italy
df2 = links_t_IT.resample('W').mean()/1e3
df2.plot(ax=ax)

#looking at the generation from solar rooftop in Italy
df = n.generators_t.p
df = df.loc["2013 01 01":"2013 12 31", ["IT0 0 solar rooftop"]].resample('D').mean()/1e3
df.plot(ax=ax)
load = n.loads_t.p.loc["2013 01 01":"2013 12 31", ["IT0 0"]].resample('D').mean()/1e3
load.plot(ax=ax)
ax.legend(['Distribution grid','Load','Solar Rooftop'])
ax.legend(['Load','Solar Rooftop','Distribution grid'],bbox_to_anchor=(0., 1.02, 1., .08), loc='upper center',
           ncol=3, borderaxespad=0.)
ax.set_xlabel('')
ax.set_ylabel('Load / Generation / Flow [GW]')

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'
name = r'\02_Flow_dist1'
plt.savefig(path+name,dpi=300, bbox_inches='tight')



#%% Making fourier plots for the distribution grid


flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='2'
co2_limits=['0.5', '0.2', '0']
flexs = ['elec_s_37'] 
techs=['Distribution']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice



for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        datos.loc[idx[techs, flex ,co2_limit], :] = np.array(network.links_t.p0.filter(like='distribution').sum(1))



#%% plotting the data

# Define which data you want to extract
n_years = 1         #default value for the years
co2_limit = '0.2'     #define which co2 level you want to use

df1 = np.hstack([np.array(datos.loc[idx['Distribution', flex, co2_limit], :])]*n_years)

t_sampling=1 # sampling rate, 1 data per hour
x = np.arange(1,(len(df1)+1)*n_years, t_sampling) 
y = np.hstack([np.array(df1)]*n_years)
nn = len(x)
y_fft=np.fft.fft(y)/nn #n for normalization    
frq=np.arange(0,1/t_sampling,1/(t_sampling*nn))        
period=np.array([1/f for f in frq])


plt.figure(2,figsize=(10,5))
plt.semilogx(period[1:nn//2],abs(y_fft[1:nn//2])**2/np.max(abs(y_fft[1:nn//2])**2),color = 'orange',label = 'Distribution grid')
plt.xticks([1, 10, 100, 1000, 10000],size = 14)
plt.yticks(size = 14)
#plt.set_xlabel(['1', '10', '100', '1000', '10000'])
plt.xlabel('cycling period (hours)',size = 12)
plt.ylabel('Nomalized frequency',size = 12)
plt.title('FFT plot of cycling frequency - CO2 limit = '+str(co2_limit),size = 16,color = 'black')
plt.axvline(x=24, color='lightgrey', linestyle='--')
plt.axvline(x=24*7, color='lightgrey', linestyle='--')
plt.axvline(x=24*30, color='lightgrey', linestyle='--')
plt.axvline(x=8760, color='lightgrey', linestyle='--')
plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
plt.text(len(df1)+200, 0.95, 'year', horizontalalignment='left', color='dimgrey', fontsize=14)
plt.axis([0,10000+6000,0,1])
plt.legend(fontsize=11,loc='upper left')


# Save the figure in the selected path
name = r'\01_FFT_grid_co2_'+str(co2_limit)+'.png'
#plt.savefig(r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'+name, dpi=300,  bbox_inches='tight')
plt.savefig(path+name, dpi=300, bbox_inches='tight') 

gs1 = gridspec.GridSpec(7, 1)
gs1.update(wspace=0.05)
co2_limits=['0.5', '0.2', '0']
storage_names=['PHS'] #,'battery','H2']
dic_color={'PHS':'darkgreen'}
storage_names=['PHS'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}
#%%

dic_color={'0.5':'red','0.2':'darkgreen','0':'orange'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
co2_limits=['0.5', '0.2', '0']
for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[0+2*i:2+2*i,0])    #[4+2*i:6+2*i,0] 
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    #ax1.plot(np.arange(0,8760), datos.loc[idx['PHS', flex, float(co2_lim)], :]/np.max(datos.loc[idx['PHS', flex, float(co2_lim)], :]), 
    #         color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
    #         label='CO$_2$='+dic_label[co2_lim])
    #ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    #y = np.hstack([np.array(datos.loc[idx['Onwind', flex, float(co2_lim)], :])]*n_years)
    #y = np.hstack([np.array(datos.loc[idx['Distribution', flex, float(co2_lim)], :])]*n_years) 
    y = np.hstack([np.array(datos.loc[idx['Distribution', flex, str(co2_lim)], :])]*n_years) 
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='upper left', shadow=True,fancybox=True,prop={'size':10})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
#plt.title('Run of River Fourier power spectrum')
name = r'\Fourier_transform_dist.jpg'
plt.savefig(path+name,dpi=300,format='jpg')


