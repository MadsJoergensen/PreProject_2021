# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#%% Wind generation

omega_w = 2*math.pi/168         #Frequency for wind time series
omega_s = 2*math.pi/24          #Frequency for solar time series




# Calculation wind time series
def G_w(t):     # calculation of eta(p)
    G_w = 4*(1+np.sin(omega_w*t))
    return G_w

# Calculation solar time series
def G_s(t):     # calculation of eta(p)
    G_s = 4*(1+np.sin(omega_s*t))
    return G_s

n = 300 # number of hours
t = np.linspace(0,n,n) 

G_w = G_w(t)
G_s = G_s(t)

plt.plot(t,G_w,color='orange',label='20 kW')
#plt.figure(dpi=480)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Power generation - wind')
plt.xlabel('Time [hrs]')
plt.axis([0, 300, 0, 8.2])
plt.ylabel('Power generation [GW]')  
plt.show()


plt.plot(t,G_s)
plt.title('Power generation - solar PV')
plt.xlabel('Time [hrs]')
plt.axis([0, 300, 0, 8.2])
plt.ylabel('Power generation [GW]')  
plt.show()


#%% Calculation of the power

# Create a list of the notes and links

Link = [[0,1],
        [1,2],
        [1,3],
        [1,4],
        [2,4]]

# Calculation of the degree matrix of each node

k_1 = 1
k_2 = 4
k_3 = 2
k_4 = 1
k_5 = 2

# Average degree of the network

L = 5
N = 5

k_avg = (k_1+k_2+k_3+k_4+k_5)/N

# 7 create the degree matrix and adjacency matrix

# Degree matrix
D_ij = [[1, 0, 0, 0, 0], 
    [0, 4, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 2]]

# Adjacency matrix

A_ij = [[0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0]
        ]

#Turning the incidence matrix into an array    
A_np = np.array(A_ij) 
print(A_np)


#Calculating K^T
A_trans= (A_np.T) 
print("\n")
print(A_trans)
print("\n")
print(A_np-A_trans)
print("\n")


# A_ij = A_ij^T

print("A =", D_ij) 
print("D[1] =", D_ij[1])      # 2nd row
print("D[1][2] =", D_ij[1][2])   # 3rd element of 2nd row
print("D[0][-1] =", D_ij[0][-1])   # Last element of 1st Row

column = [];        # empty list
for row in D_ij:
  column.append(row[2])   
  
print("3rd column =", column)

# Creation of the incidence matrix

K_old = [[1, 0, 0, 0, 0],
        [-1, 1, 1, 1, 0],
        [0, 0, -1, 0, 1],
        [0, 0, -1, 0, 0],
        [0, 0, 0, -1, -1]
        ]


K = [[1, 0, 0, 0, 0],
        [-1, 1, 1, 1, 0],
        [0, -1, 0, 0, 1],
        [0, 0, -1, 0, 0],
        [0, 0, 0, -1, -1]
        ]


#%% Creation of the Laplacian matrix

result= [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         ]


# iterate through rows
for i in range(len(A_ij)):
   # iterate through columns
   for j in range(len(A_ij[0])):
       result[i][j] = D_ij[i][j] - A_ij[i][j]

for r in result:
   print(r)
  
#Turning the incidence matrix into an array    
K_np = np.array(K) 

#Calculating K^T
K_trans= (K_np.T) 

#Calculating the Laplace matrix by L = KK^T
Laplace = np.matmul(K_np, K_trans)
print("\n")
print(Laplace)
#%% Check that the definitions match


# Printing the result from the degree matrix minus the adjacency matrix
print(np.array(result))    
print("\n")
# Printing the result from the incidence matrix times the transposed incidence matrix KK^T
print(Laplace)

# Check that they both calculate the Laplace matrix
Check_L = np.array(result)-Laplace
print("\n")
print(Check_L)

