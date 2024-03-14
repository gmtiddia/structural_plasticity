import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parametri delle distribuzioni gaussiane
media1 = 2.0
media2 = 4.0
varianza1 = 0.6
varianza2= 1.2
lim_y=0.8
# Calcola il punto medio tra le medie delle due distribuzioni
font_legend=15
punto_medio = (media1 + media2) / 2
font_label=15
plt.figure(1)
# Crea la curva della distribuzione gaussiana
x = np.linspace(-1, 8, 1000)
pdf1 = norm.pdf(x, media1, varianza1)
pdf2 = norm.pdf(x, media2, varianza2)
#plt.figure(figsize=(10,9))
#plt.rcParams['axes.linewidth'] = 1.2 # Spessore del box
# Colora l'area dei FP e FN
x_fill = np.linspace(punto_medio, 8, 1000)
x_fill2 = np.linspace(0, punto_medio, 1000)
plt.fill_between(x_fill, norm.pdf(x_fill, media1, varianza1), color='yellow', alpha=0.2, label='FPR')
plt.fill_between(x_fill2, norm.pdf(x_fill2, media2, varianza2), color='red', alpha=0.2, label='FNR')
# Plot distribuzioni gaussiane
plt.plot(x, pdf1, color='gold', label=r'$S_{\text{b}}$ distribution',linewidth=2.0)
plt.plot(x, pdf2, color='brown',label=r'$S_{\text{c}}$ distribution',linewidth=2.0)
plt.ylim(0,lim_y)
plt.axvline(x=media1, ymin=0, ymax=max(pdf1)/lim_y, color='black', linestyle='--',linewidth=2.0)
plt.axvline(x=media2, ymin=0, ymax=max(pdf2)/lim_y,color='black', linestyle='--',linewidth=2.0)
plt.axvline(x=(media1+media2)/2, ymin=0, ymax=norm.pdf((media1+media2)/2, media2, varianza2)/lim_y, color='blue', linestyle='-', label='Threshold',linewidth=2.5)

plt.text(media1-0.3, y=max(pdf1)+0.02,color='gold', fontsize=18, s=r'$\langle S_{\text{b}} \rangle $', fontweight='bold')
plt.text(media2-0.3, y=max(pdf2)+0.02, color='brown', fontsize=18, s=r'$\langle S_{\text{c}} \rangle $', fontweight='bold')
plt.xlabel(r'Signal intensity [pA $\times$ Hz]',fontsize=font_label)
plt.ylabel('Probability',fontsize=font_label)
plt.legend(fontsize=font_legend)
# Tolti ticks degli assi
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
plt.xticks([])  # Rimuovi ticks sull'asse x
plt.yticks([])  # Rimuovi ticks sull'asse y

plt.savefig('two_gaussian.png')
plt.show()
