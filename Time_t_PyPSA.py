#!/usr/bin/env python
# coding: utf-8

# # Aarhus University - Fall 2020 - Renewable Energy Systems (RES) project 
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

# We start by creating the network. In this example, the country is modelled as a single node, so the network will only include one bus.
# 
# We select the year 2015 and set the hours in that year as snapshots.
# 
# We select a country, in this case, Spain (ESP), and add one node (electricity bus) to the network.

n = pypsa.Network("elec_s_37_lv1.0__Co2L0.05-solar+p3-dist10_2030.nc")

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'


#%% Plotting time-varying data, for different technology

x  = 10    

capacity = n.generators

generators_1 = n.generators.groupby("carrier")["p_nom_opt"].sum()
generators_2 = n.generators.groupby("carrier")["p_nom_opt"].sum()

print(n.generators_t.p)


# Making the netword file into a dataframe and plotting

df = pd.DataFrame(n.generators_t.p)

df2 = df.loc["20130401":"20130714","DK0 0 onwind"]
plt.figure(figsize=(10,5),dpi=200)
ax = df2.plot(color = 'blue',label='onwind DK')
ax.set_ylabel('Capacity factor [-]')
ax.set_xlabel('Day')
ax.set_title('Solar capacity factor DK')
ax.set_ylim(0,df2.max())
ax.legend()


#%% Making fourier plots for loads/generators/storage

n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))

n.generators_t.p_max_pu.head()

#plotting the solar generation for Italy in july

#n.stores_t.e.loc["2013","DK0 0 battery"].plot(figsize=(15,3))

df = n.stores_t.e.loc["2013","DK0 0 battery"]
df1 = n.generators_t.p.loc["2013-07","DE0 0 solar"] #n.loads_t.p_set.sum(axis=1)

len(df1)

n_years=1
t_sampling=1 # sampling rate, 1 data per hour
x = np.arange(1,(len(df1)+1)*n_years, t_sampling) 
y = np.hstack([np.array(df1)]*n_years)
nn = len(x)
y_fft=np.fft.fft(y)/nn #n for normalization    
frq=np.arange(0,1/t_sampling,1/(t_sampling*nn))        
period=np.array([1/f for f in frq])


plt.figure(2,figsize=(10,5))

plt.semilogx(period[1:nn//2],abs(y_fft[1:nn//2])**2/np.max(abs(y_fft[1:nn//2])**2),color = 'black',label = 'solar PV')
plt.xticks([1, 10, 100, 1000, 10000])
#plt.set_xlabel(['1', '10', '100', '1000', '10000'])
plt.xlabel('cycling period (hours)')
plt.ylabel('Nomalized frequency')
plt.title('FFT plot of cycling frequency',size = 22,color = 'blue')
plt.legend()

# Save the figure in the selected path
name = r'\01_FFT_solar'
#plt.savefig(r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'+name, dpi=300,  bbox_inches='tight')
plt.savefig(path+name, dpi=300, bbox_inches='tight') 

#%% Fourier power series with sub-plots for solar and wind

pathdata = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\Data'


flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['Solar', 'Onwind']

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
        datos.loc[idx['Solar', flex ,co2_limit], :] = np.array(network.generators_t.p.loc["2013","DE0 0 solar"])
        datos.loc[idx['Onwind', flex, co2_limit], :] = np.array(network.generators_t.p.loc["2013","DE0 0 onwind"])
        #datos.loc[idx['Solar', flex ,co2_limit], :] = np.array(network.storage_units_t.state_of_charge[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum(axis=1)/(6*network.storage_units.p_nom[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum()))
        #datos.loc[idx['Onwind', flex, co2_limit], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'ror']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'ror']].sum())

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\Data_for_figures/storage_timeseries.pickle') 
datos.to_csv(pathdata+'\Data_for_figures/storage_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\Data_for_figures/storage_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('PHS filling level')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['PHS'] #,'battery','H2']
dic_color={'PHS':'darkgreen'}
storage_names=['PHS'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['Solar', flex, float(co2_lim)], :]/np.max(datos.loc[idx['Solar', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['Solar', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_PHS.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

## Plot for run of river
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 5))
gs1 = gridspec.GridSpec(7, 1)
gs1.update(wspace=0.05)

co2_limits=['0.5', '0.2', '0']
storage_names=['Onwind'] #,'battery','H2']
dic_color={'Onwind':'darkgreen'}
storage_names=['Onwind'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

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
    y = np.hstack([np.array(datos.loc[idx['Onwind', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_ror.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg')



#%% Fourier power series with sub-plots

pathdata = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\Data'


flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['PHS', 'ror']

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
        datos.loc[idx['Solar', flex ,co2_limit], :] = np.array(network.storage_units_t.state_of_charge[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum(axis=1)/(6*network.storage_units.p_nom[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum()))
        datos.loc[idx['Onwind', flex, co2_limit], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'ror']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'ror']].sum())

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/storage_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('PHS filling level')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['PHS'] #,'battery','H2']
dic_color={'PHS':'darkgreen'}
storage_names=['PHS'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['PHS', flex, float(co2_lim)], :]/np.max(datos.loc[idx['PHS', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['PHS', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_PHS.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

## Plot for run of river
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 5))
gs1 = gridspec.GridSpec(7, 1)
gs1.update(wspace=0.05)

co2_limits=['0.5', '0.2', '0']
storage_names=['ror'] #,'battery','H2']
dic_color={'ror':'darkgreen'}
storage_names=['ror'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

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
    y = np.hstack([np.array(datos.loc[idx['PHS', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_ror.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg')

#%% Extracting data for onshore wind for Europe

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']


flexs = ['elec_s_37'] 
techs=['Solar', 'Onwind']

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
        datos.loc[idx['Solar', flex ,co2_limit], :] = np.array(network.generators_t.p[network.generators.index[network.generators.carrier == 'solar']].sum(axis=1))
        datos.loc[idx['Onwind', flex, co2_limit], :] = np.array(network.generators_t.p[network.generators.index[network.generators.carrier == 'onwind']].sum(axis=1))


# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/storage_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',') 

# In[2]:

## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Solar capacity factor')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['Solar'] #,'battery','H2']
dic_color={'Solar':'darkgreen'}
storage_names=['Solar'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['Solar', flex, float(co2_lim)], :]/np.max(datos.loc[idx['Solar', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['Solar', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_Solar.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

## Plot for Onshore wind
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Onshore capacity factor')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

co2_limits=['0.5', '0.2', '0']
storage_names=['Onwind'] #,'battery','H2']
dic_color={'Onwind':'darkgreen'}
storage_names=['Onwind'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    #[4+2*i:6+2*i,0] 
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['Onwind', flex, float(co2_lim)], :]/np.max(datos.loc[idx['Onwind', flex, float(co2_lim)], :]), 
            color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
            label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['Onwind', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_Onwind.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg')

b = network.generators_t.p[network.generators.index[network.generators.carrier == 'solar']].sum(axis=1)                  

plt.figure()
plt.plot(np.arange(0,8760),b)

#datos.loc[idx['Solar', flex ,co2_limit], :] = np.array(network.storage_units_t.state_of_charge[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum(axis=1)/(6*network.storage_units.p_nom[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum()))
        
#%% Extracting data for H2 and battery storage

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']


flexs = ['elec_s_37'] 
techs=['H2 Store', 'Battery']

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
        datos.loc[idx['H2 Store', flex ,co2_limit], :] = np.array(network.stores_t.p[network.stores.index[network.stores.carrier == 'H2']].sum(axis=1))
        datos.loc[idx['Battery', flex, co2_limit], :] = np.array(network.stores_t.p[network.stores.index[network.stores.carrier == 'battery']].sum(axis=1))


# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/storage_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',') 

# In[2]:

## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('H2 Storage')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['H2'] #,'battery','H2']
dic_color={'H2':'darkgreen'}
storage_names=['H2'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['H2 Store', flex, float(co2_lim)], :]/np.max(datos.loc[idx['H2 Store', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['H2 Store', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_H2.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

## Plot for Onshore wind
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Battery')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

co2_limits=['0.5', '0.2', '0']
storage_names=['Battery'] #,'battery','H2']
dic_color={'Battery':'darkgreen'}
storage_names=['Battery'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    #[4+2*i:6+2*i,0] 
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['Battery', flex, float(co2_lim)], :]/np.max(datos.loc[idx['Battery', flex, float(co2_lim)], :]), 
            color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
            label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['Battery', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
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
name = r'\Fourier_transform_Battery.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg')

#datos.loc[idx['Solar', flex ,co2_limit], :] = np.array(network.storage_units_t.state_of_charge[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum(axis=1)/(6*network.storage_units.p_nom[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum()))
        


#%%  Making a double for-loop for different cases

flex= 'elec_s_37'  
lv = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-dist'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '2.0'] #, '1.2', '1.5', '2.0'

df = pd.DataFrame()

for lv in lvl:
    for co2_limit in co2_limits:
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'1'+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        df[lv+co2_limit] = generators


df1 = df.filter(like='1.0')
df2 = df.filter(like='1.1')
df3 = df.filter(like='2.0')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharey=True)
fig.suptitle('Horizontally stacked subplots')
df1.plot.bar(ax=ax1,rot=35, figsize=(12, 5) )
ax1.set_xlabel('Carrier')
ax1.set_ylabel('Installed capacity [MW]')
ax1.set_title('Carrier capacity vs. CO2 emmisions - lv1.0')
ax1.yaxis.grid()
#plt.rc('grid', linestyle="--", color='gray')
#ax1.legend(frameon = True, ncol = 5, shadow=True, bbox_to_anchor=(0.5, 1.25), loc='upper center', title = '% CO2 emmesion compared to 1990')

df2.plot.bar(ax=ax2,rot=35, figsize=(12, 5) )
ax2.set_xlabel('Carrier')
ax2.set_ylabel('Installed capacity [MW]')
ax2.set_title('Carrier capacity vs. CO2 emmisions - lv1.1')
ax2.yaxis.grid()

df3.plot.bar(ax=ax3,rot=35, figsize=(12, 5) )
ax3.set_xlabel('Carrier')
ax3.set_ylabel('Installed capacity [MW]')
ax3.set_title('Carrier capacity vs. CO2 emmisions - lv2.0')
ax3.yaxis.grid()

# In[3]

n.plot()

for c in n.iterate_components(list(n.components.keys())[2:]):
    print("Component '{}' has {} entries".format(c.name,len(c.df)))


# In[4] Temporal resolution

print(len(n.snapshots))

# In[5] Static component data

print(n.lines.head())

print(n.generators.head())

print(n.storage_units.head())

# In[6] Time-varying component data (input data)

n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))

n.generators_t.p_max_pu.head()

#plotting the solar generation for Italy in july
n.generators_t.p_max_pu.loc["2013-07","IT0 0 solar"].plot(figsize=(15,3))

#plotting the onshore wind generation for Estonia in july
n.generators_t.p_max_pu.loc["2013-07","ES0 0 onwind"].plot(figsize=(15,3))


# In[7] Total Annual System Cost

# Total system cost, in billion euroes

n.objective / 1e9

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
       line_widths=n.lines.s_nom_opt/3e3,
       line_colors=loading,
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       bus_sizes=0)
ax.axis('off');

