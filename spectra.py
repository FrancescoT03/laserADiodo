import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

length = np.linspace(350, 1000, 1000-350 +1)
for i in [12, 15, 18, 22, 24, 26, 29, 32, 34, 36, 39, 42, 46]:
    sum = 0.
    for j in range(1, 6):
        sum += np.loadtxt(f'dati_spettri/spettro_{i}_{j}.txt', unpack=True, skiprows=17, usecols=1, delimiter='\t', encoding ='latin1')
    h = sum / 5
    sigma_h = np.std([np.loadtxt(f'dati_spettri/spettro_{i}_{j}.txt', unpack=True, skiprows=17, usecols=1, delimiter='\t', encoding ='latin1') for j in range(1, 6)], axis=0) / np.sqrt(5)
       
    lambd = np.argmax(h)
    ll = np.loadtxt(f'dati_spettri/spettro_{i}_1.txt', unpack=True, skiprows=17, usecols=0, delimiter='\t', encoding ='latin1')
    plt.plot(i, ll[lambd], 'r.')

plt.xlabel('Temperature (°C)')
plt.ylabel('Wavelength (nm)')
plt.title('Wavelength vs Temperature')
plt.grid(ls='--')
plt.show()
