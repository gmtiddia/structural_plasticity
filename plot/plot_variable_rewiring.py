import os
import numpy as np
import math
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
import pandas as pd
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

    params = np.loadtxt(path+'/params.dat', dtype=str)
    dict_params = dict([[params[i][0], float(params[i][1])] for i in range(len(params))])
    
    return(dict_params)




def get_th_set(path, T):
    """
    Returns theoretical estimation of Sb, Sc and Var(Sb) given the simulation parameters.

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
    dict of the theoretical values for Sb, varSb and Sc
    """

    def erfm1(x):
        return np.sqrt(2.0)*erfinv(2.0*x - 1.0)
    
    def Phi(x):
        return 0.5*(1.0 + erf(x/np.sqrt(2.0)))
    
    def phi1(csi):
        return np.exp(-csi*csi/2.0) / np.sqrt(2.0*math.pi)

    params = get_params_dict(path)
    p = 1.0 - (1.0 - params['alpha1']*params['alpha2'])**T
    q1 = 1.0 - params['alpha1']
    q2 = 1.0 - params['alpha2']
    # average rate layer 1
    rm1 = params['alpha1']*params['nu_h_1'] + q1*params['nu_l_1']
    # average rate layer 2
    rm2 = params['alpha2']*params['nu_h_2'] + q2*params['nu_l_2']

    # rate threshold for both layers
    if(params['lognormal_rate']==1):
        sigma_ln1 = erfm1(q1) - erfm1(q1*params['nu_l_1']/rm1)
        mu_ln1 = math.log(rm1) - sigma_ln1*sigma_ln1/2.0
        yt_ln1 = erfm1(q1)*sigma_ln1 + mu_ln1
        rt1 = np.exp(yt_ln1)
        sigma_ln2 = erfm1(q2) - erfm1(q2*params['nu_l_2']/rm2)
        mu_ln2 = math.log(rm2) - sigma_ln2*sigma_ln2/2.0
        yt_ln2 = erfm1(q2)*sigma_ln2 + mu_ln2
        rt2 = np.exp(yt_ln2)
    else:
        rt1 = (params['nu_h_1'] + params['nu_l_1']) / 2.0
        rt2 = (params['nu_h_2'] + params['nu_l_2']) / 2.0

    k = p*params['C']
    # <r1**2>
    rsq1 = params['alpha1']*params['nu_h_1']*params['nu_h_1'] + (1.0 - params['alpha1'])*params['nu_l_1']*params['nu_l_1']
    # rate variance for layer 1
    if(params['lognormal_rate']==1):
        var_r1 = (np.exp(sigma_ln1*sigma_ln1) - 1.0) * np.exp(2.0*mu_ln1 + sigma_ln1*sigma_ln1)
    else:
        var_r1 = rsq1 - rm1*rm1
    
    # variance of k
    k2 = params['C']*(params['C'] - 1.0)*((1.0 - (2.0 - params['alpha1'])*params['alpha1']*params['alpha2'])**T) - params['C']*(2.0*params['C'] - 1.0)*((1.0 - params['alpha1']*params['alpha2'])**T) + params['C']*params['C']
    var_k = k2 - k*k

    # theoretical value of Sb
    Sb = params['Ws']*k*rm1 + params['Wb']*(params['C']-k)*rm1

    # theoretical value of Sc
    if(params['r']==0):
        # w/out rewiring
        Sc = params['nu_h_1']*params['Ws']*params['alpha1']*params['C'] + params['nu_l_1']*(1.0-params['alpha1'])*(params['Wb']*params['C'] + (params['Ws'] - params['Wb'])*k)
    else:
        r = params['r']
        b = (1.-(1.-params['alpha1']*params['alpha2'])**(T+r))/(1.-(1.-params['alpha1']*params['alpha2'])**r)
        av_pt = 1. - (b*r)/(T+r)
        av_kt_first = av_pt*params['C']*(1.0-params['alpha1'])
        # w/ rewiring
        Sc = params['nu_h_1']*params['Ws']*params['alpha1']*params['C'] + rm1*(1.0-params['alpha1'])*(params['Wb']*params['C'] + (params['Ws'] - params['Wb'])*k) - params['Ws']*(rm1 - params['nu_l_1'])*av_kt_first

    # if C is constant
    if(params['connection_rule']==0):
        # variance of Sb for C fixed
        varSb = (params['Ws']*params['Ws']*k + params['Wb']*params['Wb']*(params['C']-k))*var_r1 + (params['Ws'] - params['Wb'])*(params['Ws'] - params['Wb'])*rm1*rm1*var_k
    # if C is driven from Poisson distribution
    else:
        # average and variance of the Poisson distribution is C
        C_m = params['C']
        var_C = params['C']
        C2_m = params['C']*(params['C'] + 1.0)
        eta = (1.0 - params['alpha1']*params['alpha2'])**T
        csi = (1.0 - (2.0 - params['alpha1'])*params['alpha1']*params['alpha2'])**T
        # variance of Sb for C variable
        varSb = ((params['Wb'] + p*(params['Ws'] - params['Wb']))**2.0)*rm1*rm1*var_C + C_m*(p*params['Ws']*params['Ws'] + eta*params['Wb']*params['Wb'])*var_r1 + ((params['Ws'] - params['Wb'])**2)*rm1*rm1*((C2_m - C_m)*csi + C_m*eta - C2_m*eta*eta)

    # if we add noise on the test patterns
    if(params['noise_flag']==1):
        beta = params['max_noise_dev']
        Z = Phi(beta) - Phi(-beta)
        var_noise = params['rate_noise']*params['rate_noise']*(1.0 - 2.0*beta*phi1(beta)/Z)
        var_S_noise = (params['Ws']*params['Ws']*k + params['Wb']*params['Wb']*(params['C'] - k))*var_noise
        # add noise contribution to the variance of Sb
        varSb += var_S_noise

    return({"T": T, "Sb": Sb, "varSb": varSb, "Sc": Sc})


def get_th_data(path):
    """
    Returns a DataFrame w/ the theoretical estimation of Sb, Sc and Var(Sb) given the simulation parameters.

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
        r_list = [ f.name for f in os.scandir(str(path)) if f.is_dir() ]
        r_list.remove('template')
        r = [int(step) for step in r_list]
        r.sort()

        T = [ 10000 for i in range(len(r))]

        dum_Sb = []
        dum_varSb = []
        dum_Sc = []

        for i in range(len(r)):
            dum_dict = get_th_set(path+"/"+str(r[i]), T[i])
            dum_Sb.append(dum_dict['Sb'])
            dum_varSb.append(dum_dict['varSb'])
            dum_Sc.append(dum_dict['Sc'])

        dic = {"r": r, "T": T, "Sb_th": dum_Sb, "varSb_th": dum_varSb, "Sc_th": dum_Sc}

        data = pd.DataFrame(dic)
        data.to_csv(path+"/th_values.csv", index=False)
        print("csv generated")
        return(data)


def get_data(path, seeds):
    """
    Returns a dataframe w/ averaged values of Sb, Sc and Var(Sb) over the seeds.
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
        # extract t values from them
        r_list = [ f.name for f in os.scandir(str(path)) if f.is_dir() ]
        r_list.remove('template')
        r = [int(step) for step in r_list]
        r.sort()
        # define arrays to contain average and std values
        Sc_av = np.zeros(len(r)); Sc_std = np.zeros(len(r))
        Sb_av = np.zeros(len(r)); Sb_std = np.zeros(len(r))
        varSb_av = np.zeros(len(r)); varSb_std = np.zeros(len(r))
        SDNR_av = np.zeros(len(r)); SDNR_std = np.zeros(len(r))

        # extract values from data
        for step, dir in enumerate(subfolders):
            Sb_dum = []
            Sc_dum = []
            varSb_dum = []
            SDNR_dum = []
            for i in range(seeds):
                mem_out = np.loadtxt(path+"/"+str(r[step])+"/mem_out_000"+str(i)+"_0000.dat")
                Sb_dum.append(np.average(mem_out[:,1]))
                Sc_dum.append(np.average(mem_out[:,2]))
                varSb_dum.append(np.average(mem_out[:,3]))
                SDNR_dum.append(np.abs(np.average(mem_out[:,2])-np.average(mem_out[:,1]))/np.sqrt(np.average(mem_out[:,3])))
            Sc_av[step]=np.average(Sc_dum)
            Sb_av[step]=np.average(Sb_dum)
            varSb_av[step]=np.average(varSb_dum)
            SDNR_av[step]=np.average(SDNR_dum)
            Sb_std[step]=np.std(Sb_dum)
            Sc_std[step]=np.std(Sc_dum)
            varSb_std[step]=np.std(varSb_dum)
            SDNR_std[step]=np.std(SDNR_dum)
            

        
        # free some memory
        del mem_out, Sb_dum, Sc_dum, varSb_dum

        # save values on a DataFrame
        data = {"r": r, "Sb_av": Sb_av, "Sb_std": Sb_std, "varSb_av": varSb_av, "varSb_std": varSb_std, "Sc_av": Sc_av, "Sc_std": Sc_std, "SDNR_av":SDNR_av, "SDNR_std":SDNR_std}
        data = pd.DataFrame(data)
        #print(data)

        data.to_csv(path+"/values.csv", index=False)
        print("csv generated")
        return(data)


def plot_data(data, th_data):
    """
    Plot data and saves the plot

    Parameters
    ----------
    data, th_data: pandas DataFrame
        DataFrame w/ simulation and theoretical values of Sb, varSb and Sc for discrete rate model

    """

    # fontsize params
    legend_fs =20
    tick_fs = 25

    #plot discrete
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize = (20,9), constrained_layout=False, tight_layout = True)

    #subplots of discrete model
    ax1 = axs[0,0]
    ax3 = axs[1,0]
    ax5 = axs[0,1]
    ax7 = axs[1,1]

    #ax1.fill_between(data['r'][1:], data['Sb_av'][1:]-data['Sb_std'][1:], data['Sb_av'][1:]+data['Sb_std'][1:], color="blue", alpha=0.2)
    ax1.text(-0.1, 1.05, "A", weight="bold", fontsize=30, color='k', transform=ax1.transAxes)
    ax1.plot(th_data['r'], th_data['Sb_th'], "--", linewidth=2.5, color="orange", label='Theoretical estimation')
    ax1.errorbar(data['r'][1:], data['Sb_av'][1:], yerr=data['Sb_std'][1:], fmt="o", linestyle="", color="blue", label="w/ rewiring")
    ax1.errorbar(data['r'][0], data['Sb_av'][0], yerr=data['Sb_std'][0], fmt="o", markersize=10, color="red", label="w/out rewiring")
    ax1.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.set_xlabel(r"rewiring step $r$", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_ylim(1115, 1116.45)
    ax1.set_xlim(-10, 700)
    ax1.ticklabel_format(style='plain', useOffset=False, axis='y') 
    #ax1.set_xticklabels([])
    ax1.grid()
    #ax1.set_ylim(1069.6,1070.4)
    
    #ax3.fill_between(data['r'][1:], data['varSb_av'][1:]-data['varSb_std'][1:], data['varSb_av'][1:]+data['varSb_std'][1:], color="blue", alpha=0.2)
    ax3.text(-0.1, 1.05, "C", weight="bold", fontsize=30, color='k', transform=ax3.transAxes)
    ax3.plot(th_data['r'], th_data['varSb_th'], "--", linewidth=2.5, color="orange", label='Theoretical estimation')
    ax3.errorbar(data['r'][1:], data['varSb_av'][1:], fmt="o", linestyle="", yerr=data['varSb_std'][1:], color="blue", label="w/ rewiring")
    ax3.errorbar(data['r'][0], data['varSb_av'][0], yerr=data['varSb_std'][0], fmt="o", markersize=10, color="red", label="w/out rewiring")
    #ax3.plot(data['r'], th_data['varSb_th'], "--", color="red", label="Theory")
    #ax3.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax3.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax3.set_xlabel(r"rewiring step $r$", fontsize=tick_fs)
    ax3.set_xlim(-10, 700)
    ax3.tick_params(labelsize=tick_fs)
    ax3.grid()
    #ax3.set_xticklabels([])
    #ax3.legend(fontsize=legend_fs, framealpha=1.0)

    ax5.text(-0.1, 1.05, "B", weight="bold", fontsize=30, color='k', transform=ax5.transAxes)
    #ax5.fill_between(data['r'][1:], data['Sc_av'][1:]-data['Sc_std'][1:], data['Sc_av'][1:]+data['Sc_std'][1:], color="blue", alpha=0.2)
    ax5.plot(th_data['r'], th_data['Sc_th'], "--", linewidth=2.5, color="orange", label='Theoretical estimation')
    ax5.errorbar(data['r'][1:], data['Sc_av'][1:], fmt="o", linestyle="", yerr=data['Sc_std'][1:], color="blue", label="w/ rewiring")
    ax5.errorbar(data['r'][0], data['Sc_av'][0], yerr=data['Sc_std'][0], fmt="o", markersize=10, color="red", label="w/out rewiring")
    #ax5.plot(data['r'], th_data['Sc_th'], "--", color="red", label="Theory")
    #ax5.legend(title=r"$\langle S_c \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax5.set_ylabel(r"$\langle S_c \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax5.set_xlabel(r"rewiring step $r$", fontsize=tick_fs)
    ax5.set_xlim(-10, 700)
    ax5.tick_params(labelsize=tick_fs)
    ax5.grid()
    #ax5.legend(fontsize=legend_fs, framealpha=1.0)
    #ax5.set_xticklabels([])

    ax7.text(-0.1, 1.05, "D", weight="bold", fontsize=30, color='k', transform=ax7.transAxes)
    #ax7.fill_between(data['r'][1:], data['SDNR_av'][1:]-data['SDNR_std'][1:], data['SDNR_av'][1:]+data['SDNR_std'][1:], color="blue", alpha=0.2)
    ax7.plot(th_data['r'], (th_data['Sc_th']-th_data['Sb_th'])/np.sqrt(th_data['varSb_th']), "--", linewidth=2.5, color="orange", label='Theoretical estimation')
    ax7.errorbar(data['r'][1:], data['SDNR_av'][1:], fmt="o", linestyle="", yerr=data['SDNR_std'][1:], color="blue", label="w/ rewiring")
    ax7.errorbar(data['r'][0], data['SDNR_av'][0], yerr=data['SDNR_std'][0], fmt="o", markersize=10, color="red", label="w/out rewiring")
    #ax7.plot(data['r'], np.abs(th_data['Sc_th']-th_data['Sb_th'])/np.sqrt(th_data['varSb_th']), "--", color="red", label="Theory")
    ax7.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax7.set_xlabel(r"rewiring step $r$", fontsize=tick_fs)
    ax7.tick_params(labelsize=tick_fs)
    ax7.set_xlim(-10, 700)
    ax7.grid()
    ax7.legend(fontsize=legend_fs, framealpha=1.0, loc='lower right')
    #ax7.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    #ax7.legend(fontsize=legend_fs, framealpha=1.0)
    #ax7.set_xticklabels([])

   
    #fig1.subplots_adjust(wspace=0.6)
    #fig.suptitle(r'Different rewiring step, $T=5000$', fontsize=25)

    plt.savefig("variable_rewiring.png")

    plt.show()


# values extracted from simulations
data_t_variable = get_data("../simulations/variable_rewiring", 10)

# theoretical values
th_data = get_th_data("../simulations/variable_rewiring")


print("Theor")
print(th_data)

print("Sim")
print(data_t_variable)

plot_data(data_t_variable, th_data)

plt.show()
