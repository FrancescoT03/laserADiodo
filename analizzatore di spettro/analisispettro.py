import numpy as np 
from matplotlib import pyplot as plt 
import os
from scipy.optimize import curve_fit 
from scipy import signal
from scipy.optimize import least_squares
import math 




def n_picchi(x, y, n):
    i_peek , _ = signal.find_peaks(y)             # SINTASSI DI FIND_PEAKS MOLTO UTILE
    array_picchi = y[i_peek]
    indeces1 = np.arange(len(array_picchi))
    indeces2= sorted(indeces1, key=lambda i: array_picchi[i], reverse=True)[:n]
    indeces_peeks = i_peek[indeces2]
    return x[indeces_peeks] , y[indeces_peeks]

import os
with os.scandir('/home/francesco/Scrivania/labMateria/analizzatore di spettro/distanzapicchi') as d:
    for e in d:
      a1, a2, a3, a4, a5, a6 = np.genfromtxt(fname ='/home/francesco/Scrivania/labMateria/analizzatore di spettro/distanzapicchi/' + e.name, usecols=(0,1,2,3,4,5), skip_header=18,skip_footer=0, unpack=True )
      a5  = -a5
      a6 = -a6
      a7 = (a5 + a6)/2
      peakx , peaky = n_picchi(a4,a5,5)
      #print(a5,a4)
      peakxx = sorted(peakx)

      #peak2x ,peak2y = n_picchi(a4,a7,5)
      #peak3x ,peak3y = n_picchi(a4,a6,5)
      #peak2xx = sorted(peak2x)
      #print(peakxx)
      distanze = []
      for i in range (len(peakxx)-1):
            dist = peakxx[i+1] - peakxx[i]
            distanze.append(dist)

      d = distanze.pop(1)

      print(d)
      print(distanze)
      print(f'distanza picchi stesso ordine di ' + e.name)
 
      fig_1 = plt.figure(e.name)
      ax1 = fig_1.add_subplot(2,1,1) #grafico
      ax1.set_title(f'singolo ordine')
      ax2 = fig_1.add_subplot(2,1,2) #residui
      ax1.plot(a4,a5)
      ax1.plot(a4,a6)
      ax1.plot(a4,a7)
      ax1.errorbar(peakx,peaky, fmt ='.' , color='darkblue')
      #ax1.errorbar(peak2x,peak2y, fmt ='.' , color='green')
      #ax1.errorbar(peak3x,peak3y, fmt ='.' , color='red')
      for i,j in zip(peakx,peaky):
            ax1.text(i,j,f'  {i}')
      #for i,j in zip(peak2x,peak2y):
            #ax1.text(i,j,f'  {i}')
      #for i,j in zip(peak3x,peak3y):
            #ax1.text(i,j,f'  {i}')
      ax2.plot(a1,a3)
      ax1.grid(which='both', ls='dashed', color='gray')
      ax2.grid(which='both', ls='dashed', color='gray')
      ax1.set_ylabel('voltage [mV]')
      ax2.set_xlabel('time [s]')
      ax2.set_ylabel('voltage [V]')
plt.show()
