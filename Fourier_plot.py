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

n = pypsa.Network("elec_s_37_lv1.0__Co2L0-solar+p3-dist10_2030.nc")

#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'


#%%

test = n.buses_t.p

test_df1 = test.filter(like='low voltage').sum(1)

test = n.links_t.p0

test_df2 = test.filter(like='distribution').sum(1)

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
plt.title('FFT plot of cycling frequency - CO2 limit = '+str(co2_limit),size = 16,color = 'blue')
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


