# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:50:24 2021

@author: Mads Jorgensen
"""

#make the code as Python 3 compatible as possible
from __future__ import print_function, division, absolute_import

import pypsa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

network = pypsa.Network("elec_s_37_lv1.0__Co2L0.05-solar+p3-dist10_2030.nc")

#matplotlib inline

fig,ax = plt.subplots(1,1,subplot_kw={"projection":ccrs.PlateCarree()})

fig.set_size_inches(6,6)

load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

network.plot(bus_sizes=0.001*load_distribution,ax=ax,title="Load distribution")