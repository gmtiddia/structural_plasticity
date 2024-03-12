import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parametri delle distribuzioni gaussiane
media1 = 2.0
media2 = 4.0
varianza1 = 0.6
varianza2= 1.2
# Calcola il punto medio tra le medie delle due distribuzioni
font_lb=10
punto_medio = (media1 + media2) / 2
font_label=13
# Crea la curva della distribuzione gaussiana
x = np.linspace(-1, 8, 1000)
pdf1 = norm.pdf(x, media1, varianza1)
pdf2 = norm.pdf(x, media2, varianza2)

plt.rcParams['axes.linewidth'] = 1.2 # Spessore del box

# Colora l'area dei FP e FN
x_fill = np.linspace(punto_medio, 8, 1000)
x_fill2 = np.linspace(0, punto_medio, 1000)
plt.fill_between(x_fill, norm.pdf(x_fill, media1, varianza1), color='yellow', alpha=0.5, label='FPR')
plt.fill_between(x_fill2, norm.pdf(x_fill2, media2, varianza2), color='red', alpha=0.5, label='FNR')
# Plot distribuzioni gaussiane
plt.plot(x, pdf1, color='gold', label='Background signal distribution')
plt.plot(x, pdf2, color='brown',label='Coding signal distribution')
plt.axvline(x=media1, ymin=0, ymax=max(pdf1), color='black', linestyle='--')
plt.axvline(x=media2, ymin=0, ymax=max(pdf2),color='black', linestyle='--')
plt.axvline(x=(media1+media2)/2, ymin=0, ymax=norm.pdf((media1+media2)/2, media2, varianza2), color='blue', linestyle='-', label='Threshold')

plt.text(media1-0.2, y=max(pdf1)+0.02,color='gold', fontsize=12, s=r'$\langle S_b \rangle $')
plt.text(media2-0.2, y=max(pdf2)+0.02, color='brown', fontsize=12, s=r'$\langle S_c \rangle $')
plt.xlabel(r'Signal intensity [pA $\times$ Hz]',fontsize=font_label)
plt.ylabel('Probability',fontsize=font_label)
plt.legend()
# Tolti ticks degli assi
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
plt.xticks([])  # Rimuovi ticks sull'asse x
plt.yticks([])  # Rimuovi ticks sull'asse y
plt.ylim(0,1)

plt.savefig('two_gaussian.png')
plt.show()
