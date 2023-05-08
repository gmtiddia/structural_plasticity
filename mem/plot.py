import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

path = ""
directories = ["T10", "T50", "T100", "T500", "T1000"]
T = [10, 50, 100, 500, 1000]
run = 10
legend_fs = 15
tick_fs = 15

# theoretical values
S2_t = np.zeros(len(T))
Sb_t = np.zeros(len(T))
sigma2S_t = np.zeros(len(T))

# experimental values
S2_exp = np.zeros(len(T)); S2_exp_std = np.zeros(len(T))
Sb_exp = np.zeros(len(T)); Sb_exp_std = np.zeros(len(T))
sigma2S_exp = np.zeros(len(T)); sigma2S_exp_std = np.zeros(len(T))

for t,dir in enumerate(directories):
    # theoretical values are seed independent
    data = np.loadtxt(dir+"/dum_teor.txt")
    S2_t[t] = data[2]
    Sb_t[t] = data[3]
    sigma2S_t[t] = data[4]

    # now we start a loop over the seeds
    # we average the results over the T tests
    # and then we average the outcome over the "run" simulations
    S2_dum = []
    Sb_dum = []
    sigma2S_dum = []
    for i in range(run):
        data = np.loadtxt(dir+"/data"+str(i)+"/mem_out_000"+str(i)+".dat")
        S2_dum.append(np.average(data[:,2]))
        Sb_dum.append(np.average(data[:,1]))
        sigma2S_dum.append(np.average(data[:,3]))
    
    S2_exp[t]=np.average(S2_dum)
    Sb_exp[t]=np.average(Sb_dum)
    sigma2S_exp[t]=np.average(sigma2S_dum)
    S2_exp_std[t]=np.std(S2_dum)
    Sb_exp_std[t]=np.std(Sb_dum)
    sigma2S_exp_std[t]=np.std(sigma2S_dum)

# free some memory
del data

SNR_t = (S2_t-Sb_t)/(sigma2S_t**0.5); SNR_exp = (S2_exp-Sb_exp)/(sigma2S_exp**0.5)

# plots

gs = gridspec.GridSpec(2, 4)
fig = plt.figure(1, figsize = (16,9), tight_layout = True)

ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:])
ax3 = plt.subplot(gs[1, :2])
ax4 = plt.subplot(gs[1, 2:])

ax1.fill_between(T, S2_exp-S2_exp_std, S2_exp+S2_exp_std, color="red", alpha=0.2)
ax1.plot(T, S2_exp, "-", color="red", label="Simulation")
ax1.plot(T, S2_t, "--", color="blue", label="Theory")
ax1.set_xlabel("T (training examples)", fontsize=tick_fs)
ax1.set_ylabel(r"S2 [pA $\times$ Hz]", fontsize=tick_fs)
ax1.tick_params(labelsize=tick_fs)
ax1.grid()
ax1.legend(title="Signal S2", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)

ax2.fill_between(T, Sb_exp-Sb_exp_std, Sb_exp+Sb_exp_std, color="red", alpha=0.2)
ax2.plot(T, Sb_exp, "-", color="red", label="Simulation")
ax2.plot(T, Sb_t, "--", color="blue", label="Theory")
ax2.set_xlabel("T (training examples)", fontsize=tick_fs)
ax2.set_ylabel(r"Sb [pA $\times$ Hz]", fontsize=tick_fs)
ax2.tick_params(labelsize=tick_fs)
ax2.grid()
ax2.legend(title="Signal Sb", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)

ax3.fill_between(T, sigma2S_exp-sigma2S_exp_std, sigma2S_exp+sigma2S_exp_std, color="red", alpha=0.2)
ax3.plot(T, sigma2S_exp, "-", color="red", label="Simulation")
ax3.plot(T, sigma2S_t, "--", color="blue", label="Theory")
ax3.set_xlabel("T (training examples)", fontsize=tick_fs)
ax3.set_ylabel(r"$\sigma^{2}_{Sb} \quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
ax3.tick_params(labelsize=tick_fs)
ax3.grid()
ax3.legend(title=r"Variance of signal Sb", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)

ax4.plot(T, SNR_exp, "-", color="red", label="Simulation")
ax4.plot(T, SNR_t, "--", color="blue", label="Theory")
ax4.set_xlabel("T (training examples)", fontsize=tick_fs)
ax4.set_ylabel(r"$\mathrm{SNR}=\frac{S2-Sb}{\sigma _{Sb}}$", fontsize=tick_fs)
ax4.tick_params(labelsize=tick_fs)
ax4.grid()
ax4.legend(title=r"SNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


plt.savefig("memory_capacity.png")

plt.show()




