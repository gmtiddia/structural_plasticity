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
        # w/out rewiring
        S2 = params['rh1']*params['Wc']*params['p1']*params['C'] + params['rl1']*(1.0-params['p1'])*(params['W0']*params['C'] + (params['Wc'] - params['W0'])*k)
    else:
        # w/ rewiring
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
    Returns a DataFrame w/ the theoretical estimation of Sb, S2 and Var(Sb) given the simulation parameters.

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
        t_list = [ f.name for f in os.scandir(str(path)) if f.is_dir() ]
        t_list.remove('template')
        t = [int(step) for step in t_list]
        t.sort()

        T = [ 5000 for i in range(len(t))]

        dum_Sb = []
        dum_varSb = []
        dum_S2 = []

        for i in range(len(t)):
            dum_dict = get_th_set(path+"/0", T[i])
            dum_Sb.append(dum_dict['Sb'])
            dum_varSb.append(dum_dict['varSb'])
            dum_S2.append(dum_dict['S2'])

        dic = {"t": t, "T": T, "Sb_th": dum_Sb, "varSb_th": dum_varSb, "S2_th": dum_S2}

        data = pd.DataFrame(dic)
        data.to_csv(path+"/th_values.csv", index=False)
        print("csv generated")
        return(data)


def get_data(path, seeds):
    """
    Returns a dataframe w/ averaged values of Sb, S2 and Var(Sb) over the seeds.
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
        t_list = [ f.name for f in os.scandir(str(path)) if f.is_dir() ]
        t_list.remove('template')
        t = [int(step) for step in t_list]
        t.sort()
        # define arrays to contain average and std values
        S2_av = np.zeros(len(t)); S2_std = np.zeros(len(t))
        Sb_av = np.zeros(len(t)); Sb_std = np.zeros(len(t))
        varSb_av = np.zeros(len(t)); varSb_std = np.zeros(len(t))
        CNR_av = np.zeros(len(t)); CNR_std = np.zeros(len(t))

        # extract values from data
        for step, dir in enumerate(subfolders):
            Sb_dum = []
            S2_dum = []
            varSb_dum = []
            CNR_dum = []
            for i in range(seeds):
                mem_out = np.loadtxt(path+"/"+str(t[step])+"/mem_out_000"+str(i)+"_0000.dat")
                Sb_dum.append(np.average(mem_out[:,1]))
                S2_dum.append(np.average(mem_out[:,2]))
                varSb_dum.append(np.average(mem_out[:,3]))
                CNR_dum.append(np.abs(np.average(mem_out[:,2])-np.average(mem_out[:,1]))/np.sqrt(np.average(mem_out[:,3])))
            S2_av[step]=np.average(S2_dum)
            Sb_av[step]=np.average(Sb_dum)
            varSb_av[step]=np.average(varSb_dum)
            CNR_av[step]=np.average(CNR_dum)
            Sb_std[step]=np.std(Sb_dum)
            S2_std[step]=np.std(S2_dum)
            varSb_std[step]=np.std(varSb_dum)
            CNR_std[step]=np.std(CNR_dum)
            

        
        # free some memory
        del mem_out, Sb_dum, S2_dum, varSb_dum

        # save values on a DataFrame
        data = {"t": t, "Sb_av": Sb_av, "Sb_std": Sb_std, "varSb_av": varSb_av, "varSb_std": varSb_std, "S2_av": S2_av, "S2_std": S2_std, "CNR_av":CNR_av, "CNR_std":CNR_std}
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
        DataFrame w/ simulation and theoretical values of Sb, varSb and S2 for discrete rate model

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

    ax1.fill_between(data['t'][1:], data['Sb_av'][1:]-data['Sb_std'][1:], data['Sb_av'][1:]+data['Sb_std'][1:], color="blue", alpha=0.2)
    ax1.plot(data['t'][1:], data['Sb_av'][1:], "-", color="blue", label="w/ recombination")
    ax1.plot(data['t'][0], data['Sb_av'][0], "o", markersize=10, color="red", label="w/out recombination")
    #ax1.plot(data['t'], th_data['Sb_th'], "--", color="red", label="Theory")
    ax1.legend(fontsize=legend_fs, framealpha=1.0)
    ax1.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.set_xlabel(r"Recombination step t", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_ylim(1069.4, 1070.5)
    ax1.ticklabel_format(style='plain', useOffset=False, axis='y') 
    #ax1.set_xticklabels([])
    ax1.grid()
    
    ax3.fill_between(data['t'][1:], data['varSb_av'][1:]-data['varSb_std'][1:], data['varSb_av'][1:]+data['varSb_std'][1:], color="blue", alpha=0.2)
    ax3.plot(data['t'][1:], data['varSb_av'][1:], "-", color="blue", label="w/ recombination")
    ax3.plot(data['t'][0], data['varSb_av'][0], "o", markersize=10, color="red", label="w/out recombination")
    #ax3.plot(data['t'], th_data['varSb_th'], "--", color="red", label="Theory")
    ax3.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax3.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax3.set_xlabel(r"Recombination step t", fontsize=tick_fs)
    ax3.tick_params(labelsize=tick_fs)
    ax3.grid()
    #ax3.set_xticklabels([])
    ax3.legend(fontsize=legend_fs, framealpha=1.0)


    ax5.fill_between(data['t'][1:], data['S2_av'][1:]-data['S2_std'][1:], data['S2_av'][1:]+data['S2_std'][1:], color="blue", alpha=0.2)
    ax5.plot(data['t'][1:], data['S2_av'][1:], "-", color="blue", label="w/ recombination")
    ax5.plot(data['t'][0], data['S2_av'][0], "o", markersize=10, color="red", label="w/out recombination")
    #ax5.plot(data['t'], th_data['S2_th'], "--", color="red", label="Theory")
    ax5.legend(title=r"$\langle S_2 \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax5.set_ylabel(r"$\langle S_2 \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax5.set_xlabel(r"Recombination step t", fontsize=tick_fs)
    ax5.tick_params(labelsize=tick_fs)
    ax5.grid()
    ax5.legend(fontsize=legend_fs, framealpha=1.0)
    #ax5.set_xticklabels([])


    ax7.fill_between(data['t'][1:], data['CNR_av'][1:]-data['CNR_std'][1:], data['CNR_av'][1:]+data['CNR_std'][1:], color="blue", alpha=0.2)
    ax7.plot(data['t'][1:], data['CNR_av'][1:], "-", color="blue", label="w/ recombination")
    ax7.plot(data['t'][0], data['CNR_av'][0], "o", markersize=10, color="red", label="w/out recombination")
    #ax7.plot(data['t'], np.abs(th_data['S2_th']-th_data['Sb_th'])/np.sqrt(th_data['varSb_th']), "--", color="red", label="Theory")
    ax7.set_ylabel(r"CNR", fontsize=tick_fs)
    ax7.set_xlabel(r"Recombination step t", fontsize=tick_fs)
    ax7.tick_params(labelsize=tick_fs)
    ax7.grid()
    #ax7.legend(title=r"CNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=1.0)
    ax7.legend(fontsize=legend_fs, framealpha=1.0)
    #ax7.set_xticklabels([])

   
    #fig1.subplots_adjust(wspace=0.6)
    #fig.suptitle(r'Different recombination step, $T=5000$', fontsize=25)

    plt.savefig("t_study.png")

    plt.show()


# values extracted from simulations
data_t_variable = get_data("simulations/t_study_w_noise", 10)

# theoretical values
th_data = get_th_data("simulations/t_study_w_noise")


print("Theor")
print(th_data)

print("Sim")
print(data_t_variable)

plot_data(data_t_variable, th_data)

plt.show()