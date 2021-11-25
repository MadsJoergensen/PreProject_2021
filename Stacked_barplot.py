# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:52:57 2021

@author: Mads Jorgensen
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = pypsa.Network("elec_s_37_lv1.0__Co2L0-solar+p3-dist10_2030.nc")

#%% Basic plotting data

data = n.generators         #define which network component that needs to be analysed
#data = n.stores

country = ['DK0','AL0','DE0','BA0','FR0']       #define the different countries you want to check
column = 'p_nom_opt'

d = {}      # define a dictionary to save values
result = pd.DataFrame()     #define a dataframe for plotting
    
for i in country:
      
    df = pd.DataFrame(data, columns = [column]).filter(like = i,axis=0) #filtering operation of the data
    df = df.rename(columns={column: i})                                 #rename the coulums to the itterator name
    df.index = df.index.map(lambda x: str(x)[6:])                       #remove digits in the index for plotting
    
    d[i] = df       #save the dataframe in the dictionary

    result = pd.concat([result,d.get(i)], axis=1)       #adding the data to the result dataframe    
df1 = result.T  #transposing the data for plotting

#%% Stacked bar plot
#Specify the path where to store the plots
path = r'C:\Users\Mads Jorgensen\OneDrive - Aarhus Universitet\Dokumenter\3. Semester Kandidat\01_PreProject\LateX\Pictures'
  
colors={"offwind-ac": "#6895dd",
        "offwind-dc": "#74c6f2", 
        "onwind": "#235ebc",
        "solar": "#f9d002",
        "solar rooftop": '#ffef60',
        "ror": '#78AB46',
        "PHS": "g",
        "Gas": "brown"}

plt.figure()
df1.plot.bar(stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             );
plt.title('Stacked bar plot',y=0.9)
plt.ylabel('Installed capacity [MW]')
#plt.xticks(rotation=00)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
name = r'\02_generators.png'
plt.savefig(path+name,dpi=300,format='jpg', bbox_inches='tight')

#%% Making a double loop to create df's

n = pypsa.Network("elec_s_37_lv1.0__Co2L0-solar+p3-dist10_2030.nc")

#%% Basic plotting data

data = n.generators
#data = n.stores

country = ['DK0','AL0','DE0','BA0','FR0']       #define the different countries you want to check
co2 = ['0.5','0.2','0.1','0.05','0']

column = 'p_nom_opt'

d = {}      # define a dictionary to save values
result = pd.DataFrame()     #define a dataframe for plotting
co2_0 = pd.DataFrame() 
#elec_s_37_lv1.0__Co2L0.1-solar+p3-dist0.5_2030.nc

for co2_limit in co2:
    network_name= ('elec_s_37_lv1.0__Co2L'+co2_limit+'-solar+p3-dist1_2030.nc')        
    print(network_name)
    n = pypsa.Network(network_name)
    data = n.generators
    result = pd.DataFrame()
    
    for i in country:
      
        df = pd.DataFrame(data, columns = [column]).filter(like = i,axis=0) #filtering operation of the data
        df = df.rename(columns={column: i})                                 #rename the coulums to the itterator name
        df.index = df.index.map(lambda x: str(x)[6:])                       #remove digits in the index for plotting
    
        d[co2_limit,i] = df       #save the dataframe in the dictionary

        result = pd.concat([result,d[co2_limit,i]], axis=1)       #adding the data to the result dataframe
        
    if co2_limit == '0':
        co2_0 = result
    elif co2_limit == '0.05':
        co2_0p05 = result
    elif co2_limit == '0.1':
        co2_0p1 = result
    elif co2_limit == '0.2':
        co2_0p2 = result
    else:
        co2_0p5 = result
    
    
#%% plotting the looped data

plt.figure()
co2_0p05.T.plot.bar(stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             );
plt.title('Stacked bar plot',y=0.9)
plt.ylabel('Installed capacity [MW]')
#plt.xticks(rotation=00)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
name = r'\stacked.png'
plt.savefig(path+name,dpi=300,format='jpg', bbox_inches='tight')

x = d['0',i]

df1 = result.T  #transposing the data for plotting


#%% Plotting subplots 

plt.figure(1)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,figsize=(20,8),dpi = 200,sharey=True)
fig.subplots_adjust(wspace=0.0, hspace=0) #adjusting the distance between plots
co2_0p5.T.plot.bar(ax = ax1,
             stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             legend = False,
             );
ax1.set_xlabel('Country')
ax1.set_ylabel('Installed capacity [MW]')
ax1.set_title('Generator capcity - co2 lvl = 50%')
ax1.yaxis.grid()
#plt.rc('grid', linestyle="--", color='gray')
#ax1.legend(frameon = True, ncol = 5, shadow=True, bbox_to_anchor=(0.5, 1.25), loc='upper center', title = '% CO2 emmesion compared to 1990')

co2_0p2.T.plot.bar(ax = ax2,
             stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             legend = False,
             );
ax2.set_xlabel('Country')
ax2.set_ylabel('Installed capacity [MW]')
ax2.set_title('Generator capcity - co2 lvl = 20%')
ax2.yaxis.grid()

co2_0p1.T.plot.bar(ax = ax3,
             stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             legend = False,
             );
ax3.set_xlabel('Country')
ax3.set_ylabel('Installed capacity [MW]')
ax3.set_title('Generator capcity - co2 lvl = 10%')
ax3.yaxis.grid()

co2_0p05.T.plot.bar(ax = ax4,
             stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             legend = False,
             );
#ax4.set_xlabel('Country')
ax4.set_title('Generator capcity - co2 lvl = 5%')
ax4.yaxis.grid()

co2_0.T.plot.bar(ax = ax5,
             stacked=True,
             width=0.8,         #width of the bars
             rot=0,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             legend = False,
             );
#ax3.set_xlabel('Country')
#ax3.set_ylabel('Installed capacity [MW]')
ax5.set_title('Generator capcity - co2 lvl = 0%')
ax5.yaxis.grid()



legend = result.index
fig.suptitle('Installed capacity vs. CO2 constrain and Transmission expansion - Dist=1',
             y = 1,
             fontsize = 15)

fig.legend(legend,bbox_to_anchor=(0.,0.85, 1., .102), 
           loc='upper center', 
           ncol=len(legend), 
           columnspacing = 8, 
           borderaxespad=0.) #mode="expand"

#%% Looping over data - creating the arrays

onwind = np.zeros((5,5))
offwind_dc = np.zeros((5,5))
offwind_ac = np.zeros((5,5))
solar_pv = np.zeros((5,5))
solar_roof = np.zeros((5,5))
gas =  np.zeros((5,5))


#%% Extracting the data for the whole country

data = n.generators_t.p         #define which network component that needs to be analysed

data = data.sum(axis=0).sum()

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
     

#%%Making a multi index df

#Define the index we want to plot for
dist = '1'# ['0.1','0.5','1','2','10']  # '1'
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '1.2', '1.5', '2.0']

#Getting the generators from renewable energi
generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
#Getting the capacity from gas technologies
gas_tech = n.links.groupby("carrier")["p_nom_opt"].sum().filter(like='GT')
#appending the two strings
df = generators.append(gas_tech)
#Taking out the gas from generators
generator = df.drop(labels=['gas'])



datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=dist, name='Dist',),
                                                       pd.Series(data=lvl, name='lvl',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,len(generator)), name='hour',))
idx = pd.IndexSlice

#rename coulmn index
j = 0
index = generator.index
for i in index:
    
    datos =datos.rename(columns={datos.columns[j]: str(i)})
    j = j+1

#looping over the values
for co2_limit in co2_limits:
        network_name= ('elec_s_37_' + 'lv'+ '1.0' + '__Co2L' +co2_limit+ '-' + 'solar+p3' +'-'+'dist'+dist+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name)
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        #Getting the capacity from gas technologies
        gas_tech = n.links.groupby("carrier")["p_nom_opt"].sum().filter(like='GT')
        #appending the two strings
        df = generators.append(gas_tech)
        #Taking out the gas from generators
        generator = df.drop(labels=['gas'])
        #adding them to the DataFrame datos
        datos.loc[idx[dist, lvl ,co2_limit], :] = np.array(generator)




#%% Plotting the results

colors={"offwind-ac": "#6895dd",
        "offwind-dc": "#74c6f2", 
        "onwind": "#235ebc",
        "solar": "#f9d002",
        "solar rooftop": '#ffef60',
        "ror": '#78AB46',
        "PHS": "g",
        "OCGT": "wheat",
        "CCGT": '#ee8340'
        }
plt.figure()
datos.plot.bar(stacked=True,
             width=0.8,         #width of the bars
             rot=90,             #rotation of the x-axis  
             alpha=1,           #transparancy
             color=colors,      #color scheme defines above
             edgecolor='black',  #edgecolor of the bars
             );
plt.title('Stacked bar plot',y=0.9)
plt.ylabel('Installed capacity [MW]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)

#%%


for i in country:
      
    df = pd.DataFrame(data, columns = [column]).filter(like = i,axis=0) #filtering operation of the data
    df = df.rename(columns={column: i})                                 #rename the coulums to the itterator name
    df.index = df.index.map(lambda x: str(x)[6:])                       #remove digits in the index for plotting
    
    d[i] = df       #save the dataframe in the dictionary

    result = pd.concat([result,d.get(i)], axis=1)       #adding the data to the result dataframe    
df1 = result.T  #transposing the data for plotting
