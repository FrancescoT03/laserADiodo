import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

#import dei dati
I_1, W_1 = np.genfromtxt(fname ='/home/francesco/Scrivania/I_P_roomtemperature_up.txt', usecols=(0,1), skip_header=0,skip_footer=0, unpack=True )
I_2, W_2 = np.genfromtxt(fname ='/home/francesco/Scrivania/I_P_12C(1).txt', usecols=(0,1), skip_header=0,skip_footer=0, unpack=True )
I_3, W_3 = np.genfromtxt(fname ='/home/francesco/Scrivania/I_P_44_down.txt', usecols=(0,2), skip_header=0,skip_footer=0, unpack=True )
#calcolo di derivate prime e seconde
d_1 = np.gradient(W_1, I_1)
dd_1 = np.gradient(d_1,I_1)

d_2 = np.gradient(W_2, I_2)
dd_2 = np.gradient(d_2,I_2)

d_3 = np.gradient(W_3, I_3)
dd_3 = np.gradient(d_3,I_3)

epsilon = 10 #definizione di un treshold per la derivata seconda

mask1 = dd_1 - epsilon
mask2 = dd_2 - epsilon
mask3 = dd_3 - epsilon


#plot dei dati
fig_1 = plt.figure('temperatura ambiente')
ax1 = fig_1.add_subplot(1,1,1)
ax1.errorbar(I_1,W_1, fmt='.', color = 'k')
ax1.errorbar(I_1[mask1<0],W_1[mask1<0], fmt='.', color = 'blue')
ax1.errorbar(I_1,d_1, fmt='.', color = 'red')
ax1.errorbar(I_1,dd_1, fmt='.',color = 'orange')
ax1.plot(I_1, np.full(I_1.shape, 0), linestyle='--')
#plt.show()

fig_2 = plt.figure('temperatura 12 C')
ax1 = fig_2.add_subplot(1,1,1)
ax1.errorbar(I_2,W_2, fmt='.', color = 'k')
ax1.errorbar(I_2[mask2<0],W_2[mask2 <0], fmt='.', color = 'blue')
ax1.errorbar(I_2,d_2, fmt='.', color = 'red')
ax1.errorbar(I_2,dd_2, fmt='.',color = 'orange')
ax1.plot(I_2, np.full(I_2.shape, 0), linestyle='--')
#plt.show()

fig_3 = plt.figure('temperatura 44 C')
ax1 = fig_3.add_subplot(1,1,1)
ax1.errorbar(I_3,W_3, fmt='.', color = 'k')
ax1.errorbar(I_3[mask3 <0],W_3[mask3 <0], fmt='.', color = 'blue')
ax1.errorbar(I_3,d_3, fmt='.', color = 'red')
ax1.errorbar(I_3,dd_3, fmt='.',color = 'orange')
ax1.plot(I_3, np.full(I_3.shape, 0), linestyle='--')
plt.show()

epsilon = 0.5 #definizione di un treshold per la derivata seconda



