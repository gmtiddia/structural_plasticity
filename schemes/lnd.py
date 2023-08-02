import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
import math
import random

pi = math.pi

x=np.linspace(0.001,10,10000);

def erfm1(x):
    return np.sqrt(2.0)*erfinv(2.0*x - 1.0)

#questi sono i valori reali della nostra simulazione
sigma_log= 1.12
mu_log=0.0894999

#valori inventati per r_t r_h e r_l, giusto per rendere intuitivo il
#concetto

x_t=4
x_l=0.65
x_h=6.2

lognormale= 1.0/np.sqrt((2*pi))* np.exp(-(np.log(x-mu_log)**2)/(2.0*sigma_log**2))/(x*sigma_log)

def lognormal(x):
    return(1.0/np.sqrt((2*pi))* np.exp(-(np.log(x-mu_log)**2)/(2.0*sigma_log**2))/(x*sigma_log))

plt.figure(1)
plt.plot(x, lognormale, '-', color='black')
plt.vlines(x_t, 0, lognormal(x_t), colors = "green", linestyles='solid', linewidth=3.0)
plt.vlines(x_h, 0, lognormal(x_h), colors = "blue", linestyles='solid', linewidth=2.0)
plt.vlines(x_l, 0, lognormal(x_l), colors = "red", linestyles='solid', linewidth=2.0)
plt.fill_between(np.linspace(0, x_t, x_t*10001//10), lognormal(np.linspace(0, x_t, x_t*10001//10)), color='red', alpha=0.1)
plt.fill_between(np.linspace(x_t, 10, (10-x_t)*10001//10), lognormal(np.linspace(x_t, 10, (10-x_t)*10001//10)), color='blue', alpha=0.1)

plt.text(x_t, lognormal(x_t)+0.025, r"$\nu_t$", color = "green", fontsize=18, fontweight='bold')
plt.text(x_h, lognormal(x_h)+0.025, r"$\nu_h$", color = "blue", fontsize=18, fontweight='bold')
plt.text(x_l, lognormal(x_l)+0.025, r"$\nu_{\ell}$", color = "red", fontsize=18, fontweight='bold')

plt.xticks([], [])
plt.yticks([], [])
plt.xlabel(r"$\nu$ [spikes/s]", fontsize=15)
plt.ylabel(r"$\rho(\nu)$", fontsize=15)

plt.ylim(0,0.575)
plt.savefig('lognormal_dist.png')










plt.show()








