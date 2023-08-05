import os
import numpy as np
import math
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
        T = [int(t) for t in T_list]
        T.sort()

        dum_Sb = []
        dum_varSb = []
        dum_S2 = []

        for i in range(len(T)):
            dum_dict = get_th_set(path+"/500", T[i])
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


def plot_data(discr, th_discr, ln, th_ln, ln_noise, th_ln_noise):
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
    legend_fs = 15
    tick_fs = 15

    fig, axs = plt.subplots(ncols=2, nrows=4, figsize = (11,15), tight_layout = True)
    ax1 = axs[0,0] # discrete rate - Sb
    ax2 = axs[0,1] # lognormal rate - Sb
    ax3 = axs[1,0] # discrete rate - varSb
    ax4 = axs[1,1] # lognormal rate - varSb
    ax5 = axs[2,0] # discrete rate - S2
    ax6 = axs[2,1] # lognormal rate - S2
    ax7 = axs[3,0] # discrete rate - CNR
    ax8 = axs[3,1] # lognormal rate - CNR

    ax1.set_title("Discrete rate model", fontsize=legend_fs)
    ax1.fill_between(discr['T'], discr['Sb_av']-discr['Sb_std'], discr['Sb_av']+discr['Sb_std'], color="red", alpha=0.2)
    ax1.plot(discr['T'], discr['Sb_av'], "-", color="blue", label="Simulation")
    ax1.plot(discr['T'], th_discr['Sb_th'], "--", color="red", label="Theory")
    #ax1.set_xlabel("T training patterns", fontsize=tick_fs)
    ax1.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    #ax1.set_xscale('log')
    ax1.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax1.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax1.grid()
    #ax1.legend(title=r"$\langle S_b \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)

    ax2.set_title("Continuous rate model", fontsize=legend_fs)
    #ax2.fill_between(ln['T'], ln['Sb_av']-ln['Sb_std'], ln['Sb_av']+ln['Sb_std'], color="blue", alpha=0.2)
    ax2.plot(ln['T'], ln['Sb_av'], "-", color="blue", label="Simulation")
    ax2.plot(ln['T'], th_ln['Sb_th'], "--", color="red", label="Theory")

    #ax2.fill_between(ln_noise['T'], ln_noise['Sb_av']-ln_noise['Sb_std'], ln_noise['Sb_av']+ln_noise['Sb_std'], color="cornflowerblue", alpha=0.2)
    ax2.plot(ln_noise['T'], ln_noise['Sb_av'], "-", color="cornflowerblue", label="Simulation - noise")
    ax2.plot(ln_noise['T'], th_ln_noise['Sb_th'], "--", color="orange", label="Theory - noise")

    #ax2.set_xlabel("T training patterns", fontsize=tick_fs)
    ax2.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax2.tick_params(labelsize=tick_fs)
    #ax2.set_xscale('log')
    ax2.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax2.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax2.grid()
    #ax2.legend(title=r"$\langle S_b \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax2.legend(fontsize=legend_fs, framealpha=1.0)


    #ax3.fill_between(discr['T'], discr['varSb_av']-discr['varSb_std'], discr['varSb_av']+discr['varSb_std'], color="blue", alpha=0.2)
    ax3.plot(discr['T'], discr['varSb_av'], "-", color="blue", label="Simulation")
    ax3.plot(discr['T'], th_discr['varSb_th'], "--", color="red", label="Theory")
    #ax3.set_xlabel("T training patterns", fontsize=tick_fs)
    ax3.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax3.tick_params(labelsize=tick_fs)
    #ax3.set_xscale('log')
    ax3.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax3.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax3.grid()
    #ax3.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


    #ax4.fill_between(ln['T'], ln['varSb_av']-ln['varSb_std'], ln['varSb_av']+ln['varSb_std'], color="blue", alpha=0.2)
    ax4.plot(ln['T'], ln['varSb_av'], "-", color="blue", label="Simulation")
    ax4.plot(ln['T'], th_ln['varSb_th'], "--", color="red", label="Theory")

    #ax4.fill_between(ln_noise['T'], ln_noise['varSb_av']-ln_noise['varSb_std'], ln_noise['varSb_av']+ln_noise['varSb_std'], color="cornflowerblue", alpha=0.2)
    ax4.plot(ln_noise['T'], ln_noise['varSb_av'], "-", color="cornflowerblue", label="Simulation - noise")
    ax4.plot(ln_noise['T'], th_ln_noise['varSb_th'], "--", color="orange", label="Theory - noise")

    #ax4.set_xlabel("T training patterns", fontsize=tick_fs)
    ax4.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax4.tick_params(labelsize=tick_fs)
    #ax4.set_xscale('log')
    ax4.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax4.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax4.grid()
    #ax4.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


    #ax5.fill_between(discr['T'], discr['S2_av']-discr['S2_std'], discr['S2_av']+discr['S2_std'], color="blue", alpha=0.2)
    ax5.plot(discr['T'], discr['S2_av'], "-", color="blue", label="Simulation")
    ax5.plot(discr['T'], th_discr['S2_th'], "--", color="red", label="Theory")
    #ax5.set_xlabel("T training patterns", fontsize=tick_fs)
    ax5.set_ylabel(r"$\langle S_2 \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax5.tick_params(labelsize=tick_fs)
    #ax5.set_xscale('log')
    ax5.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax5.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax5.grid()
    #ax5.legend(title=r"$\langle S_2 \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


    #ax6.fill_between(ln['T'], ln['S2_av']-ln['S2_std'], ln['S2_av']+ln['S2_std'], color="blue", alpha=0.2)
    ax6.plot(ln['T'], ln['S2_av'], "-", color="blue", label="Simulation")
    ax6.plot(ln['T'], th_ln['S2_th'], "--", color="red", label="Theory")

    #ax6.fill_between(ln_noise['T'], ln_noise['S2_av']-ln_noise['S2_std'], ln_noise['S2_av']+ln_noise['S2_std'], color="cornflowerblue", alpha=0.2)
    ax6.plot(ln_noise['T'], ln_noise['S2_av'], "-", color="cornflowerblue", label="Simulation - noise")
    ax6.plot(ln_noise['T'], th_ln_noise['S2_th'], "--", color="orange", label="Theory - noise")

    #ax6.set_xlabel("T training patterns", fontsize=tick_fs)
    ax6.set_ylabel(r"$\langle S_2 \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax6.tick_params(labelsize=tick_fs)
    #ax6.set_xscale('log')
    ax6.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax6.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax6.grid()
    #ax6.legend(title=r" $\langle S_2 \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


    ax7.plot(discr['T'], np.abs(discr['S2_av']-discr['Sb_av'])/np.sqrt(discr['varSb_av']), "-", color="blue", label="Simulation")
    ax7.plot(discr['T'], np.abs(th_discr['S2_th']-th_discr['Sb_th'])/np.sqrt(th_discr['varSb_th']), "--", color="red", label="Theory")
    ax7.set_xlabel("T training patterns", fontsize=tick_fs)
    ax7.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax7.tick_params(labelsize=tick_fs)
    #ax7.set_xscale('log')
    ax7.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax7.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax7.grid()
    #ax5.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


    ax8.plot(ln['T'], np.abs(ln['S2_av']-ln['Sb_av'])/np.sqrt(ln['varSb_av']), "-", color="blue", label="Simulation")
    ax8.plot(ln['T'], np.abs(th_ln['S2_th']-th_ln['Sb_th'])/np.sqrt(th_ln['varSb_th']), "--", color="red", label="Theory")

    ax8.plot(ln_noise['T'], np.abs(ln_noise['S2_av']-ln_noise['Sb_av'])/np.sqrt(ln_noise['varSb_av']), "-", color="cornflowerblue", label="Simulation - noise")
    ax8.plot(ln_noise['T'], np.abs(th_ln_noise['S2_th']-th_ln_noise['Sb_th'])/np.sqrt(th_ln_noise['varSb_th']), "--", color="orange", label="Theory - noise")

    ax8.set_xlabel("T training patterns", fontsize=tick_fs)
    ax8.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax8.tick_params(labelsize=tick_fs)
    #ax8.set_xscale('log')
    ax8.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax8.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    ax8.grid()
    #ax8.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)


    plt.savefig("discrete_vs_continuous.png")

    
def plot_data_luca(discr, th_discr, ln, th_ln, ln_noise, th_ln_noise):
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
    legend_fs = 21
    tick_fs = 25

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize = (20, 9), tight_layout = True)
    ax1 = axs[0,0] # discrete rate - Sb
    ax2 = axs[0,1] # lognormal rate - Sb
    ax3 = axs[1,0] # discrete rate - varSb
    ax4 = axs[1,1] # lognormal rate - varSb


    ax1.plot(discr['T'], discr['Sb_av'], "-", color="blue", label="Discrete rate")
    ax1.plot(ln['T'], ln['Sb_av'], "--", color="red", label="Continuous rate")
    ax1.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.set_xlabel("T training patterns", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_xscale('log')
    #ax1.legend(title=r"$\langle S_b \rangle$", title_fontsize=legend_fs, fontsize=legend_fs, framealpha=1.0)
    ax1.legend(fontsize=legend_fs, framealpha=1.0)

    ax2.plot(discr['T'], discr['S2_av'], "-", color="blue", label="Discrete rate")
    ax2.plot(ln['T'], ln['S2_av'], "--", color="red", label="Continuous rate")
    ax2.set_ylabel(r"$\langle S_2 \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax2.set_xlabel("T training patterns", fontsize=tick_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.set_xscale('log')
    #ax2.legend(title=r"$\langle S_2 \rangle$", title_fontsize=legend_fs, fontsize=legend_fs, framealpha=1.0)
    ax2.legend(fontsize=legend_fs, framealpha=1.0)


    ax3.plot(discr['T'], discr['varSb_av'], "-", color="blue", label="Discrete rate")
    ax3.plot(ln['T'], ln['varSb_av'], "--", color="red", label="Continuous rate")
    ax3.set_xlabel("T training patterns", fontsize=tick_fs)
    ax3.set_ylabel(r"$\sigma^2_b \quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax3.tick_params(labelsize=tick_fs)
    ax3.set_xscale('log')
    #ax3.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax3.legend(fontsize=legend_fs, framealpha=1.0)


    ax4.plot(discr['T'], np.abs(discr['S2_av']-discr['Sb_av'])/np.sqrt(discr['varSb_av']), "-", color="blue", label="Discrete rate")
    ax4.plot(ln['T'], np.abs(ln['S2_av']-ln['Sb_av'])/np.sqrt(ln['varSb_av']), "--", color="red", label="Continuous rate")
    ax4.set_xlabel("T training patterns", fontsize=tick_fs)
    ax4.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax4.tick_params(labelsize=tick_fs)
    ax4.set_xscale('log')
    #ax4.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax4.legend(fontsize=legend_fs, framealpha=1.0)



    plt.savefig("discrete_vs_continuous.png")










# values extracted from simulations
discr_rate = get_data("simulations/discr_rate_simulations", 10)
ln_rate = get_data("simulations/no_noise_simulations", 10)
ln_rate_noise = get_data("simulations/noise_1Hz_simulations", 10)

# theoretical values
th_discr = get_th_data("simulations/discr_rate_simulations")
th_ln = get_th_data("simulations/no_noise_simulations")
th_ln_noise = get_th_data("simulations/noise_1Hz_simulations")


print("Theor")
print(th_discr)

print("Sim")
print(discr_rate)


plot_data_luca(discr_rate, th_discr, ln_rate, th_ln, ln_rate_noise, th_ln_noise)

plt.show()
