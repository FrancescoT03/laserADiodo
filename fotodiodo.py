import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy

#import dei dati
I_1, sigma_I1,W_1, sigma_W1 = np.genfromtxt(fname ='/home/francesco/Scrivania/labMateria/laserADiodo/I_P_roomtemperature_up.txt', usecols=(0,1,2,3), skip_header=0,skip_footer=0, unpack=True )
#I_1, sigma_I1, W_1, sigma_W1 = np.genfromtxt(fname ='/home/francesco/Scrivania/labMateria/laserADiodo/I_P_12C(1).txt', usecols=(0,1,2,3), skip_header=0,skip_footer=0, unpack=True )
#I_1,sigma_I1,W_1, sigma_W1 = np.genfromtxt(fname ='/home/francesco/Scrivania/labMateria/laserADiodo/I_P_44_down.txt', usecols=(0,1,2,3), skip_header=0,skip_footer=0, unpack=True )

"""
# questo serve solo per il terzo set di dati che è orientato in ordine decrescente
I_1 = np.flip(I_1)
sigma_I1 = np.flip(sigma_I1)
W_1 = np.flip(W_1)
sigma_W1 = np.flip(sigma_W1)
#"""

#definizione della temperatura per nome file
temp = 12

#calcolo di derivate prime e seconde
d_1 = np.gradient(W_1, I_1)
dd_1 = np.gradient(d_1,I_1)

#####
#definizione di un treshold per la derivata seconda
epsilon = 10
#####
#definizione delle maschere di selezione per i nuovi array

diff_1  = dd_1 - epsilon #la differenza sarà maggiore di zero solo per i punti nel gomito, diminuire il treshold aumenta i punti nel gomito
elbow_1 = I_1[diff_1 >0] # definisco il gomito come i punti per cui la derivata seconda è maggiore del treshold
mask_1_left = np.where(I_1 < elbow_1[0], True,False) #definisco una maschera per i punti prima del gomito
mask_1_right= np.where(I_1 > elbow_1[-1],True,False) #deginisco una maschera per i punti dopo del gomito

'''
#print degli array solo per vedere se funziona
print(I_1)
print(elbow_1)
print(mask_1_left)
print(mask_1_right)
print(I_1[mask_1_left])
print(I_1[mask_1_right])
#'''

#definisco la retta di Fit.
def line(x,m, q):
    y = m*x + q
    return y
#definisco la funzione punto di intersezione fra le due rette calcolata analiticamente
#la funzione prende in input due array che devono contenere coefficiente angolare e offset in questo ordine
#l[0] è il coefficiente angolare l[1] è l'offset
def point(l1, l2):
    x = (l2[1] - l1[1])/(l1[0] - l2[0])
    y = (l1[0]*l2[1] - l2[0]*l1[1])/(l1[0] - l2[0])
    return np.array([x, y])


#Fitting
popt_1_right, pcov_1_right = curve_fit(line,I_1[mask_1_right ],W_1[mask_1_right], sigma=sigma_W1[mask_1_right]) #retta destra
popt_1_left, pcov_1_left = curve_fit(line,I_1[mask_1_left],W_1[mask_1_left], sigma=sigma_W1[mask_1_left])      #retta sinistra

#calcolo dei residui
res_1_left= (W_1[mask_1_left] - line(I_1[mask_1_left], *popt_1_left))/sigma_W1[mask_1_left]
res_1_right= (W_1[mask_1_right] - line(I_1[mask_1_right], *popt_1_right))/sigma_W1[mask_1_right]
#calcolo incertezze sui popt
sigma_1_left = np.sqrt(pcov_1_left.diagonal())
sigma_1_right = np.sqrt(pcov_1_right.diagonal())

#calcolo dei chi quadri
#ne calcolo 2 perché sono fit a priori separati ma non credo cambi tanto perché il chi quadro totale è la somma dei due in ogni caso
chi2_left = np.sum(res_1_left**2)
chi2_right = np.sum(res_1_right**2)
print(chi2_left, chi2_right)

#calclo il punto di intersezione
#uso uncertanties per propagare l'errore sui parametri stimati dal fit
P = point(unumpy.uarray(popt_1_left,sigma_1_left), unumpy.uarray(popt_1_right,sigma_1_right))
print(f'punto di intersezione fra le due rette {P}')
P_x = -popt_1_right[1]/popt_1_right[0]
sigma_Px = P_x*np.sqrt(np.sum((sigma_1_right/popt_1_right)**2))
print(f'punto di intersezione fra la retta sopra soglia e asse x {P_x} +- {sigma_Px}')
#plot dei dati
X = np.linspace(I_1[0], I_1[-1], 100) #linspace per plot dati
fig_1 = plt.figure(f'temperatura {temp} °C')
ax1 = fig_1.add_subplot(3,1,1) #grafico
ax1.set_title(f'temperatura {temp} °C')
ax3 = fig_1.add_subplot(3,1,3) #derivate
ax2 = fig_1.add_subplot(3,1,2) #residui

ax1.errorbar(elbow_1,W_1[diff_1 >0], fmt='.', color = 'gray') #il gomito scartato lo plotto di grigio
ax1.errorbar(I_1[mask_1_left],W_1[mask_1_left], yerr=sigma_W1[mask_1_left], fmt='.', color = 'blue') #sinistra
ax1.errorbar(I_1[mask_1_right],W_1[mask_1_right], yerr= sigma_W1[mask_1_right],fmt='.', color = 'purple') #destra
ax1.plot(X,line(X,*popt_1_left), color ='blue') #fit sinistra
ax1.plot(X,line(X,*popt_1_right), color = 'purple') #fit destra
ax1.errorbar(P[0].n, P[1].n, fmt='.', color = 'darkgreen') #punto di intersezione
ax1.vlines(P[0].n,W_1[0],W_1[-1],color='darkgreen', linestyles='-.')
#bellurie
ax1.set_ylabel('potenza [$\mu W$]')
ax1.set_ylim(min(W_1) - 50)  # numero completamente arbitrario usato solo perché sembrava bello

#plot dei residui
ax2.errorbar(I_1[mask_1_left], res_1_left, fmt='.', color='blue') #sinistra 
ax2.errorbar(I_1[mask_1_right], res_1_right, fmt='.', color = 'purple') #destra
#ax2.plot(I_1, np.full(I_1.shape, 0), linestyle='--', color ='k') #linea 0
ax2.set_ylabel('residui normalizzati')

#plot delle derivate
ax3.plot(I_1,d_1, marker='.',color = 'red', label = 'derivata prima')
ax3.plot(I_1,dd_1, marker='.',color = 'orange', label ='derivata seconda') 
#ax3.plot(I_1, np.full(I_1.shape, 0), linestyle='--', color ='k') #zero
ax3.plot(I_1, np.full(I_1.shape, epsilon), linestyle='--', color = 'darkgreen', label='treshold') #treshold
ax3.legend(handletextpad=1,fancybox=True,loc='upper left',labelcolor='black',fontsize = 12)
ax3.set_xlabel('corrente [$mA$]')

#griglie
ax1.grid(which='both', ls='dashed', color='gray')
ax2.grid(which='both', ls='dashed', color='gray')
ax3.grid(which='both', ls='dashed', color='gray')
plt.show()