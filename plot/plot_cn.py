import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import math

path = "../simulations/C_vs_N_data/"
path_10k = path+"cn/"
path_1k = path+"cn1000/"
path_2k = path+"cn2000/"
path_5k = path+"cn5000/"
#path_bkg = "background/"
directories = [ "T10000",  "T50000",  "T100000", "T200000"]
T = [10000,  50000,  100000, 200000]
R10K=[1000/10000,  2000/10000,  5000/10000, 10000/10000]
R50K =[1000/50000,  2000/50000,  5000/50000, 10000/50000]
R100K= [1000/100000,  2000/100000,  5000/100000, 10000/100000]
R200K= [1000/200000,  2000/200000,  5000/200000, 10000/200000]
run = [1000/200000,  2000/200000,  5000/200000, 10000/200000]
legend_fs = 20
tick_fs = 25

# theoretical values
Sb_t = np.zeros(len(T))
Sb_t_5=np.zeros(len(T))
Sb_t_2=np.zeros(len(T))
Sb_t_1=np.zeros(len(T))
# simulation values w/10000 synapses
varSb_exp = np.zeros(len(T)); Sb_exp_std = np.zeros(len(T))

# simulation values w/5000 synapses
varSb_exp_5 = np.zeros(len(T)); Sb_exp_std_5 = np.zeros(len(T))

# simulation values w/2000 synapses                                                                                             
varSb_exp_2 = np.zeros(len(T)); Sb_exp_std_2 = np.zeros(len(T))

# simulation values w/1000 synapses                                                                                             
varSb_exp_1 = np.zeros(len(T)); Sb_exp_std_1 = np.zeros(len(T))

# simulation values from background
#Sb_exp_bkg = np.zeros(len(T)); Sb_exp_std_bkg = np.zeros(len(T))
#Sb_variance= np.zeros(len(T))



for t,dir in enumerate(directories):
    # theoretical values are seed independent
    data = np.loadtxt(path_10k+dir+"/dum_teor.txt")
    data_5k= np.loadtxt(path_5k+dir+"/dum_teor.txt")
    data_2k= np.loadtxt(path_2k+dir+"/dum_teor.txt")
    data_1k= np.loadtxt(path_1k+dir+"/dum_teor.txt")
    Sb_t[t] = data[3]
    Sb_t_5[t] = data_5k[3]
    Sb_t_2[t]= data_2k[3]
    Sb_t_1[t]= data_1k[3]
    # now we start a loop over the seeds
    # we average the results over the T tests
    # and then we average the outcome over the "run" simulations
    
    Sb_dum = []; Sb_dum_5 = []; Sb_dum_2=[]; Sb_dum_1=[];
    for i in [0]:

        data = np.loadtxt(path_10k+dir+"/mem_out_000"+str(i)+".dat")
        data_5k = np.loadtxt(path_5k+dir+"/mem_out_000"+str(i)+".dat")
        data_2k = np.loadtxt(path_2k+dir+"/mem_out_000"+str(i)+".dat")
        data_1k = np.loadtxt(path_1k+dir+"/mem_out_000"+str(i)+".dat")
       # data_bkg = np.loadtxt(path_bkg+dir+"/bkg_out_000"+str(i)+".dat") 


        Sb_dum.append(np.average(data[:,3]))
        Sb_dum_5.append(np.average(data_5k[:,3]))
        Sb_dum_2.append(np.average(data_2k[:,3]))
        Sb_dum_1.append(np.average(data_1k[:,3]))

    

    varSb_exp[t]=np.average(Sb_dum)
    varSb_exp_5[t]=np.average(Sb_dum_5)
    varSb_exp_2[t]=np.average(Sb_dum_2)
    varSb_exp_1[t]=np.average(Sb_dum_1)

    
   # Sb_exp_std[t]=np.std(Sb_dum)
   # Sb_exp_std_5[t]=np.std(Sb_dum_5)
   # Sb_exp_std_2[t]=np.std(Sb_dum_2)
   # Sb_exp_std_1[t]=np.std(Sb_dum_1)

    #  Sb_exp_bkg[t]=np.average(Sb_dum_bkg)
  #  Sb_exp_std_bkg[t]=np.std(Sb_dum_bkg)
  #  Sb_variance[t]=np.average(sigma2S_bkg)
    

# plots
gs = gridspec.GridSpec(1, 2)
fig = plt.figure(1, figsize = (20,9), tight_layout = True)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])



C_frac_10 =[]
C_frac_50 =[]
C_frac_100 =[]
C_frac_200 =[]
N=[10000, 50000, 100000, 200000]
varSb_t_100k = []
varSb_t_10k = []
varSb_t_50k = []
varSb_t_200k = []

var_t = []
S2_t = []
for j in [0,1,2,3]: 
     for i in range(1,10001,1):

          T = 1000
          alpha1 = 1.0e-3
          alpha2 = 1.0e-3;
          C = i
          Wb = 0.1
          Ws = 1.0
          q1 = 1.0 - alpha1
          rl = 2.0
          rh = 50.0

          p = 1.0 - np.power(1.0 - alpha1*alpha2, T)
          r = alpha1*rh + (1.0 - alpha1)*rl
          sigma_ln1=1.12015;
          mu_ln1=0.0894999;
          k = p*C
          k2 = C*(C - 1)*np.power(1.0 - (2.0 - alpha1)*alpha1*alpha2, T) - C*(2*C - 1)*np.power(1.0 - alpha1*alpha2, T) + C*C
          sigma2k = k2 - k*k

          sigma2r = (math.exp(sigma_ln1*sigma_ln1) -1.0)* math.exp(2.0*mu_ln1 + sigma_ln1*sigma_ln1)
          k=p*C
          k2 = C*(C - 1)*np.power(1.0 - (2.0 - alpha1)*alpha1*alpha2, T) - C*(2*C - 1)*np.power(1.0 - alpha1*alpha2, T) + C*C
          
          vart = (Ws*Ws*k + Wb*Wb*(C-k))*sigma2r + (Ws - Wb)*(Ws - Wb)*r*r*sigma2k
          sigma2k = k2 - k*k
          Sbt = Ws*k*r + Wb*(C-k)*r

          if j==0:
              varSb_t_10k.append(vart)
              C_frac_10.append(C/N[j])
          if j==1:
              varSb_t_50k.append(vart)
              C_frac_50.append(C/N[j])

          if j==2:
              varSb_t_100k.append(vart)
              C_frac_100.append(C/N[j])
          if j==3:
              varSb_t_200k.append(vart)
              C_frac_200.append(C/N[j])
#S2t = rh*Ws*alpha1*C + rl*(1.0-alpha1)*(Wb*C + (Ws - Wb)*k)
#vart = (Ws*Ws*k + Wb*Wb*(C-k))*sigma2r + (Ws - Wb)*(Ws - Wb)*r*r*sigma2k

ax1.text(-0.1, 1.05, "A", weight="bold", fontsize=30, color='k', transform=ax1.transAxes)
ax1.plot(R10K, [varSb_exp_1[0], varSb_exp_2[0], varSb_exp_5[0], varSb_exp[0] ], "o", linewidth=2, color="blue", label="Sim N=10K")
ax1.plot(R50K, [varSb_exp_1[1], varSb_exp_2[1], varSb_exp_5[1], varSb_exp[1]], "o", linewidth=2, color="red", label="Sim N=50K")
ax1.plot(R100K, [varSb_exp_1[2], varSb_exp_2[2], varSb_exp_5[2], varSb_exp[2]], "o", linewidth=2, color="black", label="Sim N=100K")
ax1.plot(R200K, [varSb_exp_1[3], varSb_exp_2[3], varSb_exp_5[3], varSb_exp[3]], "o", linewidth=2, color="green", label="Sim N=200K")
ax1.plot(C_frac_10,varSb_t_10k, "-.", color="blue", linewidth=2, label="Th N=10K")
ax1.plot(C_frac_50,varSb_t_50k, "--", color="red", linewidth=2, label="Th N=50K")
ax1.plot(C_frac_100,varSb_t_100k, "-", color="black", linewidth=2, label="Th N=100K")
ax1.plot(C_frac_200,varSb_t_200k, ":", color="green", linewidth=2, label="Th N=200K")
ax1.set_xscale('log')
ax1.set_xlim(0.001,1.1)


ax1.set_xlabel(r'$\mathcal{C}/\mathcal{N}$ ratio', fontsize=tick_fs)
ax1.set_ylabel(r"$\sigma_{b}^2$ [$pA^2$ $\times$ $Hz^2$]", fontsize=tick_fs)
#ax1.set_ylabel(r"$\frac{S2-S2_{th}}{S2_{th}}$ (%)  ", fontsize=tick_fs)
ax1.tick_params(labelsize=tick_fs)
#ax1.grid()
#ax1.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.5)
ax1.legend(fontsize=legend_fs, framealpha=0.5)

ax2.text(-0.1, 1.05, "B", weight="bold", fontsize=30, color='k', transform=ax2.transAxes)
ax2.plot(R10K, [abs(varSb_exp_1[0]-varSb_t_10k[999])/varSb_t_10k[999]*100, abs(varSb_exp_2[0]-varSb_t_10k[1999])/varSb_t_10k[1999]*100,abs( varSb_exp_5[0]-varSb_t_10k[4999])/varSb_t_10k[4999]*100, abs(varSb_exp[0]-varSb_t_10k[9999])/varSb_t_10k[9999]*100 ], "-.", color="blue", linewidth=2, label="N=10K")
ax2.plot(R50K, [abs(varSb_exp_1[1]-varSb_t_50k[999])/varSb_t_50k[999]*100, abs(varSb_exp_2[1]-varSb_t_50k[1999])/varSb_t_50k[1999]*100,abs( varSb_exp_5[1]-varSb_t_50k[4999])/varSb_t_50k[4999]*100, abs(varSb_exp[1]-varSb_t_50k[9999])/varSb_t_50k[9999]*100 ], "--", color="red", linewidth=2, label="N=50K")
ax2.plot(R100K,[abs((varSb_exp_1[2]-varSb_t_100k[999])/varSb_t_100k[999]*100),abs( (varSb_exp_2[2]-varSb_t_100k[1999])/varSb_t_100k[1999]*100),abs( (varSb_exp_5[2]-varSb_t_100k[4999])/varSb_t_100k[4999]*100), abs((varSb_exp[2]-varSb_t_100k[9999])/varSb_t_100k[9999]*100)], "-", color="black", linewidth=2, label="N=100K")
#ax2.plot(T200K,[Sb_exp_1[0]-Sb_t_10k[999], Sb_exp_2[0]-Sb_t_10k[1999], Sb_exp_5[0]-Sb_t_10k[4999], Sb_exp[0]-Sb_t_10k[9999], "--", color="green", label="Teoria N=200K")
#ax2.plot(C_frac_10,[Sb_exp_1[1]-Sb_t_50k[999], Sb_exp_2[1]-Sb_t_50k[1999], Sb_exp_5[1]-Sb_t_50k[4999], Sb_exp[1]-Sb_t_50k[9999] ], "--", color="blue", label="Teoria N=50K")
ax2.plot(R200K, [abs(varSb_exp_1[3]-varSb_t_200k[999])/varSb_t_200k[999]*100, abs(varSb_exp_2[3]-varSb_t_200k[1999])/varSb_t_200k[1999]*100,abs( varSb_exp_5[3]-varSb_t_200k[4999])/varSb_t_200k[4999]*100, abs(varSb_exp[3]-varSb_t_200k[9999])/varSb_t_200k[9999]*100 ], ":", color="green", label="N=200K")




ax2.set_xlabel(r'$\mathcal{C}/\mathcal{N}$ ratio', fontsize=tick_fs)
ax2.set_ylabel(r"Relative error $\quad$[%]", fontsize=tick_fs)                                                                    
ax2.tick_params(labelsize=tick_fs)
ax2.set_xscale('log')
#ax2.grid()
#ax2.legend(title=r"Relative error of $\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.5)
ax2.legend(fontsize=legend_fs, framealpha=0.5)


plt.savefig("cn_plot.png")
plt.show()
