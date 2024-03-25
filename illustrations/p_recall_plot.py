import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf,erfinv

# Funzione inversa della funzione errore complementare


# Genera dati per il plot
x_values = np.linspace(0.0, 7, 1000)  # Evita divisione per zero


#inversa della funzione probabilità di recall che mi fa trovare la soglia
p_thr=0.95
sdnr_thr = erfinv(2*p_thr-1)*np.sqrt(8)

#funzione probabilità di recall
y_values = 0.5 + 0.5*erf(x_values/np.sqrt(8))

#y1=erfinv(x_values/np.sqrt(8))-erfinv(x_values/(3*np.sqrt(8)))
legend_fs =13
tick_fs = 13
tick_text= 17

#plt.rcParams['axes.linewidth'] = 1.5 # Spessore del box
plt.plot(x_values, y_values, label=R'$P_C$',color='black',linewidth=1.5)
plt.hlines(y=0.95, xmin=0, xmax=sdnr_thr, color='red', linestyle='--',label=r'$P_C=95\%$')
plt.vlines(x=sdnr_thr, ymin=0.5, ymax=0.95, color='blue', linestyle=':')
plt.text(sdnr_thr, 0.55, r'$\text{SDNR}_{thr}$', color='blue', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'),fontsize=tick_fs)
plt.text(1, 0.95, r'$P_{C}$', color='red',  ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'),fontsize=tick_fs)
plt.title('')
plt.xlim(0,7)
plt.ylim(0.5,1)
plt.tick_params(axis='both', which='major', labelsize=tick_fs)
plt.xlabel(r'SDNR',fontsize=tick_fs)
plt.ylabel(r'$P_C$',fontsize=tick_fs)
plt.legend(fontsize=legend_fs)
plt.grid(False)
plt.savefig('p_recall.png')
plt.show()
