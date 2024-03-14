import os
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec


def get_params_dict(path):
    """
    Returns Python dictionary of the simulation parameters from params.dat file

    Parameters
    ----------
    path: str
        Absolute path of the folder containing the subfolders for different values of T

    Returns
    -------
    dict_params: dict
        Python dictionary containing simulation parameters
    """

    params = np.loadtxt(path+'/params.dat', dtype=np.str)
    dict_params = dict([[params[i][0], float(params[i][1])] for i in range(len(params))])
    
    return(dict_params)


def get_th_set(path, T):
    """
    Returns theoretical estimation of Sb, S2 and Var(Sb) given the simulation parameters.

    Parameters
    ----------
    path: str
        Absolute path of the folder containing the subfolders for different values of T
        Needed to get the simulation parameters
    T: int
        Number of training patterns. Needed to evaluate the theoretical values
        given the simulation parameters
    
    Returns
    -------
    dict of the theoretical values for Sb, varSb and S2
    """

    def erfm1(x):
        return np.sqrt(2.0)*erfinv(2.0*x - 1.0)
    
    def Phi(x):
        return 0.5*(1.0 + erf(x/np.sqrt(2.0)))
    
    def phi1(csi):
        return np.exp(-csi*csi/2.0) / np.sqrt(2.0*math.pi)

    params = get_params_dict(path)
    p = 1.0 - (1.0 - params['p1']*params['p2'])**T
    q1 = 1.0 - params['p1']
    q2 = 1.0 - params['p2']
    # average rate layer 1
    rm1 = params['p1']*params['rh1'] + q1*params['rl1']
    # average rate layer 2
    rm2 = params['p2']*params['rh2'] + q2*params['rl2']

    # rate threshold for both layers
    if(params['lognormal_rate']==1):
        sigma_ln1 = erfm1(q1) - erfm1(q1*params['rl1']/rm1)
        mu_ln1 = math.log(rm1) - sigma_ln1*sigma_ln1/2.0
        yt_ln1 = erfm1(q1)*sigma_ln1 + mu_ln1
        rt1 = np.exp(yt_ln1)
        sigma_ln2 = erfm1(q2) - erfm1(q2*params['rl2']/rm2)
        mu_ln2 = math.log(rm2) - sigma_ln2*sigma_ln2/2.0
        yt_ln2 = erfm1(q2)*sigma_ln2 + mu_ln2
        rt2 = np.exp(yt_ln2)
    else:
        rt1 = (params['rh1'] + params['rl1']) / 2.0
        rt2 = (params['rh2'] + params['rl2']) / 2.0

    k = p*params['C']
    # <r1**2>
    rsq1 = params['p1']*params['rh1']*params['rh1'] + (1.0 - params['p1'])*params['rl1']*params['rl1']
    # rate variance for layer 1
    if(params['lognormal_rate']==1):
        var_r1 = (np.exp(sigma_ln1*sigma_ln1) - 1.0) * np.exp(2.0*mu_ln1 + sigma_ln1*sigma_ln1)
    else:
        var_r1 = rsq1 - rm1*rm1
    
    # variance of k
    k2 = params['C']*(params['C'] - 1.0)*((1.0 - (2.0 - params['p1'])*params['p1']*params['p2'])**T) - params['C']*(2.0*params['C'] - 1.0)*((1.0 - params['p1']*params['p2'])**T) + params['C']*params['C']
    var_k = k2 - k*k

    # theoretical value of Sb
    Sb = params['Wc']*k*rm1 + params['W0']*(params['C']-k)*rm1

    # theoretical value of S2
    if(params['change_conn_step']==0):
        # without rewiring
        S2 = params['rh1']*params['Wc']*params['p1']*params['C'] + params['rl1']*(1.0-params['p1'])*(params['W0']*params['C'] + (params['Wc'] - params['W0'])*k)
    else:
        # with rewiring
        S2 = params['rh1']*params['Wc']*params['p1']*params['C'] + rm1*(1.0-params['p1'])*(params['W0']*params['C'] + (params['Wc'] - params['W0'])*k)

    # if C is constant
    if(params['connection_rule']==0):
        # variance of Sb for C fixed
        varSb = (params['Wc']*params['Wc']*k + params['W0']*params['W0']*(params['C']-k))*var_r1 + (params['Wc'] - params['W0'])*(params['Wc'] - params['W0'])*rm1*rm1*var_k
    # if C is driven from Poisson distribution
    else:
        # average and variance of the Poisson distribution is C
        C_m = params['C']
        var_C = params['C']
        C2_m = params['C']*(params['C'] + 1.0)
        eta = (1.0 - params['p1']*params['p2'])**T
        csi = (1.0 - (2.0 - params['p1'])*params['p1']*params['p2'])**T
        # variance of Sb for C variable
        varSb = ((params['W0'] + p*(params['Wc'] - params['W0']))**2.0)*rm1*rm1*var_C + C_m*(p*params['Wc']*params['Wc'] + eta*params['W0']*params['W0'])*var_r1 + ((params['Wc'] - params['W0'])**2)*rm1*rm1*((C2_m - C_m)*csi + C_m*eta - C2_m*eta*eta)

    # if we add noise on the test patterns
    if(params['noise_flag']==1):
        beta = params['max_noise_dev']
        Z = Phi(beta) - Phi(-beta)
        var_noise = params['rate_noise']*params['rate_noise']*(1.0 - 2.0*beta*phi1(beta)/Z)
        var_S_noise = (params['Wc']*params['Wc']*k + params['W0']*params['W0']*(params['C'] - k))*var_noise
        # add noise contribution to the variance of Sb
        varSb += var_S_noise

    return({"T": T, "Sb": Sb, "varSb": varSb, "S2": S2})


def get_th_data(path):
    """
    Returns a DataFrame with the theoretical estimation of Sb, S2 and Var(Sb) given the simulation parameters.

    Parameters
    ----------
    path: str
        Absolute path of the folder containing the subfolders for different values of T
    
    Returns
    -------
    data: pandas DataFrame
        DataFrame containing the theoretical values
        The DataFrame is also saved as csv file on path     
    """

    print("Getting theoretical estimations")

    if(os.path.isfile(path+"/th_values.csv")):
        print("Data already collected. Loading existing csv")
        data = pd.read_csv(path+"/th_values.csv")
        return(data)
    else:
        print("Generating csv file")
        T_list = [ f.name for f in os.scandir(str(path)) if f.is_dir() ]
        T_list.remove('template')
        #T_list.remove('500')
        #T_list.remove('1000')
        T = [int(t) for t in T_list]
        T.sort()

        dum_Sb = []
        dum_varSb = []
        dum_S2 = []

        for i in range(len(T)):
            dum_dict = get_th_set( T[i])
            dum_Sb.append(dum_dict['Sb'])
            dum_varSb.append(dum_dict['varSb'])
            dum_S2.append(dum_dict['S2'])

        dic = {"T": T, "Sb_th": dum_Sb, "varSb_th": dum_varSb, "S2_th": dum_S2}

        data = pd.DataFrame(dic)
        data.to_csv(path+"/th_values.csv", index=False)
        print("csv generated")
        return(data)


def get_data(path, seeds):
    """
    Returns a dataframe with averaged values of Sb, S2 and Var(Sb) over the seeds.
    Needs files mem_out_seed_*.dat contained in path.

    Parameters
    ----------
    path: str
        Absolute path of the folder containing the subfolders for different values of T
    seeds: int
        Number of simulations using different seed for every configuration
    
    Returns
    -------
    data: Pandas DataFrame

    """

    print("Collecting data from .dat files")

    if(os.path.isfile(path+"/values.csv")):
        print("Data already collected. Loading existing csv")
        data = pd.read_csv(path+"/values.csv")
        return(data)
    else:
        print("Generating csv file")
        # collect subfolders name
        subfolders = [ f.path for f in os.scandir(str(path)) if f.is_dir() ]
        subfolders.remove(str(path)+"/template")
        # extract T values from them
        T_list = [ f.name for f in os.scandir(str(path)) if f.is_dir() ]
        T_list.remove('template')
        T = [int(t) for t in T_list]
        T.sort()
        # define arrays to contain average and std values
        S2_av = np.zeros(len(T)); S2_std = np.zeros(len(T))
        Sb_av = np.zeros(len(T)); Sb_std = np.zeros(len(T))
        varSb_av = np.zeros(len(T)); varSb_std = np.zeros(len(T))

        # extract values from data
        for t, dir in enumerate(subfolders):
            Sb_dum = []
            S2_dum = []
            varSb_dum = []
            for i in range(seeds):
                mem_out = np.loadtxt(path+"/"+str(T[t])+"/mem_out_000"+str(i)+"_0000.dat")
                Sb_dum.append(np.average(mem_out[:,1]))
                S2_dum.append(np.average(mem_out[:,2]))
                varSb_dum.append(np.average(mem_out[:,3]))
            S2_av[t]=np.average(S2_dum)
            Sb_av[t]=np.average(Sb_dum)
            varSb_av[t]=np.average(varSb_dum)
            Sb_std[t]=np.std(Sb_dum)
            S2_std[t]=np.std(S2_dum)
            varSb_std[t]=np.std(varSb_dum)
        
        # free some memory
        del mem_out, Sb_dum, S2_dum, varSb_dum

        # save values on a DataFrame
        data = {"T": T, "Sb_av": Sb_av, "Sb_std": Sb_std, "varSb_av": varSb_av, "varSb_std": varSb_std, "S2_av": S2_av, "S2_std": S2_std}
        data = pd.DataFrame(data)
        #print(data)

        data.to_csv(path+"/values.csv", index=False)
        print("csv generated")
        return(data)


def plot_data(discr, ln,  ln_noise,  norew_noise):
    """
    Plot data and saves the plot

    Parameters
    ----------
    discr, th_discr: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and S2 for discrete rate model
    ln, th_ln: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and S2 for lognormal rate model
    ln_noise, th_ln_noise: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and S2 for lognormal rate model with noise

    """

    # fontsize params
    legend_fs = 20
    tick_fs = 25

    widths= [1, 1, 1, 1]
    heights= [3, 1, 3, 1]

    fig, axs = plt.subplots(ncols=2, nrows=4, figsize = (20,18), constrained_layout=False,gridspec_kw={'height_ratios': heights})

  
    #subplots of discrete model
    ax1 = axs[0,0] # discrete rate - Sb
    ax2 = axs[1,0] # residue - Sb
    ax3 = axs[2,0] # discrete rate - varSb
    ax4 = axs[3,0] # residue rate - varSb
    ax5 = axs[0,1] # discrete rate - S2
    ax6 = axs[1,1] # residue rate - S2
    ax7 = axs[2,1] # discrete rate - CNR
    ax8 = axs[3,1] # residue rate - CNR

    #parameters of grid
    y_begin=0.1
    x_begin=0.12
    height_res=0.125
    height_dat=0.275
    space_s=0.01
    space_l=0.065
    central_space=0.1
    h_space=0.2
    width_g=0.35


    ############grid plot discrete model##################
    
    ax1.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax2.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax3.set_position([x_begin, y_begin+height_res+space_s, width_g, height_dat])
    ax4.set_position([x_begin, y_begin, width_g, height_res])
    ax5.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax6.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax7.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s, width_g, height_dat])
    ax8.set_position([x_begin+width_g+central_space, y_begin, width_g, height_res])


    plt.figure(1)
    ax1.text(-0.1, 0.95, "A", weight="bold", fontsize=30, color='k', transform=ax1.transAxes)
    #ax1.set_title("Discrete rate model", fontsize=legend_fs)
    #ax1.fill_between(discr['T'], discr['Sb_av']-discr['Sb_std'], discr['Sb_av']+discr['Sb_std'], color="red", alpha=0.2)
    ax1.plot(ln_noise['T'], ln_noise['Sb_av'], "o", color="blue", label="w/ rewiring")
    ax1.plot(norew_noise['T'], norew_noise['Sb_av'], "^", markersize=5, color="red", label="w/out rewiring")
    #ax1.set_xlim(5000,100000)
    ax1.legend(fontsize=legend_fs, framealpha=1.0)
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax1.set_ylabel(r"$ \langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    #ax1.set_xscale('log')
    #ax1.grid()
    ax1.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax1.set_xticklabels([])
    #ax1.legend(title=r"$S_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)

  #  print((ln_noise['Sb_av']))
   #  print((norew_noise['Sb_av']))
  #  print((norew_noise['Sb_av']))
    
 #   sottr=np.subtract(ln_noise['Sb_av'],norew_noise['Sb_av'])
 #   xxx111=np.array(norew_noise['Sb_av'])
 #   xxx222=np.array(ln_noise['Sb_av'])
 #   print((xxx222-xxx111))
 #   print(xxx222)
 #  # print(((ln_noise['Sb_av'][1:21]-norew_noise['Sb_av'])))



    
    ax2.plot(ln_noise['T'], abs((np.array(ln_noise['Sb_av'])-np.array(norew_noise['Sb_av']))/np.array(ln_noise['Sb_av'])*100), "--", color="green")
    #ax2.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax2.set_ylabel(r"$\dfrac{\langle S_b \rangle - \langle S_{b} \rangle ^{nr}}{\langle S_{b}\rangle} \quad $[%] ", fontsize=tick_fs)
    ax2.set_xlabel('T training patterns', fontsize=tick_fs)
    ax2.tick_params(labelsize=tick_fs)
    #ax2.set_xscale('log')
    ax2.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax2.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    # ax2.grid()

    ax3.text(-0.1, 1, "C", weight="bold", fontsize=30, color='k', transform=ax3.transAxes)
    ax3.plot(ln_noise['T'], ln_noise['varSb_av'], "o", color="blue", label="w/ rewiring")
    ax3.plot(norew_noise['T'], norew_noise['varSb_av'],"^", markersize=5, color="red", label="w/out rewiring")
    ax3.legend(fontsize=legend_fs, framealpha=1.0)
    ax3.set_ylabel(r"$\sigma^2_{b}$ [$pA^2 \times Hz^2$]", fontsize=tick_fs)
    ax3.set_ylim(0,17000)
    ax3.tick_params(labelsize=tick_fs)
    ax3.set_xticks([5000, 25000, 50000, 75000, 100000])
    #ax3.set_xlim(5000,100000)
    ax3.set_xticklabels([])



    ax4.plot(norew_noise['T'], abs((np.array(ln_noise['varSb_av'])-np.array(norew_noise['varSb_av']))/np.array(ln_noise['varSb_av'])*100), "--", color="green")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax4.set_ylabel(r"$\dfrac{\sigma^2_{b} - \sigma_{{b}^{nr}}^{2}}{\sigma^2_{b}}\quad$[%] ", fontsize=tick_fs)
    ax4.set_xlabel('T training patterns', fontsize=tick_fs)
    ax4.tick_params(labelsize=tick_fs)
    ax4.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax4.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])

    ax5.text(-0.1, 0.95, "B", weight="bold", fontsize=30, color='k', transform=ax5.transAxes)
    ax5.plot(ln_noise['T'], ln_noise['S2_av'], "o", color="blue", label="w/ rewiring")
    ax5.plot(norew_noise['T'], norew_noise['S2_av'], "^", markersize=5, color="red", label="w/out rewiring")
    #ax5.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax5.set_ylabel(r"$\langle S_c \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax5.tick_params(labelsize=tick_fs)
    ax5.set_xticks([5000, 25000, 50000, 75000, 100000])
    #ax5.grid()
    #ax5.set_xlim(5000,100000)
    ax5.legend(fontsize=legend_fs, framealpha=1.0)
    #ax5.legend(title=r"$S_c$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax5.set_xticklabels([])

    ax6.plot(norew_noise['T'], abs((np.array(ln_noise['S2_av'])-np.array(norew_noise['S2_av']))/np.array(ln_noise['S2_av'])*100), "--", color="green")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax6.set_ylabel(r"$\dfrac{\langle S_c \rangle - \langle S_{c}\rangle ^{nr}}{\langle S_{c} \rangle}\quad$[%] ", fontsize=tick_fs)
    ax6.set_xlabel('T training patterns', fontsize=tick_fs)
    ax6.tick_params(labelsize=tick_fs)
    ax6.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax6.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])

    CNR_rew=np.array(np.abs(ln_noise['S2_av']-ln_noise['Sb_av'])/np.sqrt(ln_noise['varSb_av']))
    CNR_norew=np.array(np.abs(norew_noise['S2_av']-norew_noise['Sb_av'])/np.sqrt(norew_noise['varSb_av']))
    ax7.plot(ln_noise['T'], np.abs(ln_noise['S2_av']-ln_noise['Sb_av'])/np.sqrt(ln_noise['varSb_av']), "o", color="blue", label="w/ rewiring")
    ax7.plot(norew_noise['T'], np.abs(norew_noise['S2_av']-norew_noise['Sb_av'])/np.sqrt(norew_noise['varSb_av']), "^", markersize=5, color="red", label="w/out rewiring")
    
    prob_thr=0.95
    SDNR_thr=erfinv(2*prob_thr-1) *2*np.sqrt(2)

   # ax7.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax7.text(-0.1, 1, "D", weight="bold", fontsize=30, color='k', transform=ax7.transAxes)

    ax7.plot(np.linspace(5000,100000,5),SDNR_thr*np.ones(5), linestyle='-',color='coral')
    ax7.text(80000, y=SDNR_thr+0.2, color='coral', fontsize=17, s=r'$\text{SDNR}_{\text{thr}}$')
    ax7.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax7.tick_params(labelsize=tick_fs)
    ax7.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax7.legend(fontsize=legend_fs, framealpha=1.0)
   # ax7.grid()
    #ax7.set_xlim(5000,100000)
    ax7.set_xticklabels([])
    #ax8.legend(title=r"CNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)

    ax8.plot(norew_noise['T'], abs((CNR_rew-CNR_norew)/CNR_norew*100), "--", color="green", label="Simulation")
  #  ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax8.set_ylabel(r"$\dfrac{SDNR - SDNR^{nr}}{SDNR^{nr}}\quad$[%] ", fontsize=tick_fs)
    ax8.set_xlabel('T training patterns', fontsize=tick_fs)
    ax8.tick_params(labelsize=tick_fs)
    ax8.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax8.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])

    #fig.subplots_adjust(bottom=0.1, top=0.97, right=0.975, left=0.12, wspace = 0.25)

    plt.savefig("no_rewiring.png")










# values extracted from simulations
norew_noise = get_data("../simulations/simulation_no_rewiring_continuo",10)
discr_rate = get_data("../simulations/discr_rate_simulations", 10)
ln_rate = get_data("../simulations/no_noise_simulations", 10)
ln_rate_noise = get_data("../simulations/noise_1Hz_simulations", 10)

T = ln_rate['T']

sdnr_rew = np.asarray([(ln_rate['S2_av'][i]-ln_rate['Sb_av'][i])/np.sqrt(ln_rate['varSb_av'][i]) for i in range(len(ln_rate['T']))])
sdnr_norew = np.asarray([(norew_noise['S2_av'][i]-norew_noise['Sb_av'][i])/np.sqrt(norew_noise['varSb_av'][i]) for i in range(len(norew_noise['T']))])


def sqrt_function(X, a):
    return a / np.sqrt(X) 
prob_thr=0.95
sdnr_thr = erfinv(2*prob_thr-1)*np.sqrt(8)

paramsrew, covariance0 = curve_fit(sqrt_function, T, sdnr_rew)
paramsnorew, covariance1 = curve_fit(sqrt_function, T, sdnr_norew)

a_fit_rew = paramsrew[0]
a_fit_norew = paramsnorew[0]

T_max_rew=a_fit_rew**2/(sdnr_thr)**2
T_max_norew=a_fit_norew**2/(sdnr_thr)**2

diff_patterns=(T_max_rew-T_max_norew)/T_max_norew*100
print(T_max_rew)
print(T_max_norew)
print("Con il rewiring la rete riesce ad immagazzinare il " + str(diff_patterns)+ "\% di pattern in più")
# theoretical values
#th_norew_noise = get_th_data("simulations/simulation_no_rewiring_continuo")
#th_discr = get_th_data("simulations/discr_rate_simulations")
#th_ln = get_th_data("simulations/no_noise_simulations")
#th_ln_noise = get_th_data("simulations/noise_1Hz_simulations")

plot_data(discr_rate, ln_rate,  ln_rate_noise, norew_noise)
plt.show()
