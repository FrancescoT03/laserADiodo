# allargamento fasci gaussiani

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Theta, I = np.loadtxt('width_parallel2.txt', unpack=True, skiprows=1)
theta, i = np.loadtxt('width_perpendicular2.txt', unpack=True, skiprows=1)
plt.plot(Theta, I, '.r', label='parallel')
plt.plot(theta, i, '.g', label='perpendicular')
plt.grid()
plt.legend()

def gauss(x, A, x0, sigma, y0):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + y0
x_fit1 = np.linspace(-9,12, 1000)
x_fit2 = np.linspace(-22,30, 1000)


popt1, pcov1 = curve_fit(gauss, Theta, I, p0=[5, 2, 3, 0]) 
popt2, pcov2 = curve_fit(gauss, theta, i, p0=[5, 3, 5, 0])

plt.plot(x_fit1, gauss(x_fit1, *popt1), '-r', label=f'$\sigma_\parallel$ = {popt1[2]:.2f}')
plt.plot(x_fit2, gauss(x_fit2, *popt2), '-g', label=f'$\sigma_\perp$ = {popt2[2]:.2f}')
plt.xlabel('Angle (°)')
plt.ylabel('Current (mA)')
plt.title('Gaussian Beam Widths')
plt.legend()



plt.show()