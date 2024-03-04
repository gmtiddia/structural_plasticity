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
        #print("Old varSb: {}".format(varSb))
        varSb += var_S_noise
        #print("New varSb: {}".format(varSb))

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
            dum_dict = get_th_set(path+"/5000", T[i])
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


def plot_data(discr, th_discr, ln, th_ln, ln_noise, th_ln_noise, label, saturation,prob_thr):


    #Blocco che permette all'utente di decidere la probabilità di soglia durante il RUN
    '''
    while True:
        try:
            prob_thr = float(input(r"Inserire soglia sulla probabilità P_C: "))
            if 0 <= prob_thr <= 1:
                break  # Esci dal ciclo se il valore è valido
            else:
                print("Il numero inserito non è compreso tra 0 e 1. Riprova.")
        except ValueError:
            print("Inserisci un numero valido.")
    '''


    def SDNR_thres(prob_thr):
        
        SDNR_thr=erfinv(prob_thr) *2*np.sqrt(2)
        return(SDNR_thr)
    
    SDNR_thr=SDNR_thres(prob_thr)
    print('La soglia su SDNR corrispondente a P_C=' + str(prob_thr)+ ' è:' + str(SDNR_thr))

    """
    Plot data and saves the plot

    Parameters
    ----------
    discr, th_discr: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and S2 for discrete rate model
    ln, th_ln: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and S2 for lognormal rate model
    ln_noise, th_ln_noise: list of pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and S2 for lognormal rate model with noise

    """

    # fontsize params
    legend_fs =20
    tick_fs = 25

    #width and height ratios of subplots
    widths= [1, 1, 1, 1]
    heights= [3, 1, 3, 1]

    #plot discrete
    fig, axs = plt.subplots(ncols=2, nrows=4, figsize = (20,18), constrained_layout=False,gridspec_kw={'height_ratios': heights})

    #plot lognormal model
    #fig1, axs1 = plt.subplots(ncols=2, nrows=4, figsize = (20,18), constrained_layout=False,gridspec_kw={'height_ratios': heights})
    fig1, axs1 = plt.subplots(ncols=2, nrows=4, figsize = (20,18),constrained_layout=True,gridspec_kw={'height_ratios': heights})
  
    #subplots of discrete model
    ax1 = axs[0,0] # discrete rate - Sb
    ax2 = axs[1,0] # residue - Sb
    ax3 = axs[2,0] # discrete rate - varSb
    ax4 = axs[3,0] # residue rate - varSb
    ax5 = axs[0,1] # discrete rate - S2
    ax6 = axs[1,1] # residue rate - S2
    ax7 = axs[2,1] # discrete rate - SDNR
    ax8 = axs[3,1] # residue rate - SDNR

    #subplots of log model

    ax9 = axs1[0,0] # lognormal rate - Sb
    ax10 = axs1[1,0] # residue - Sb
    ax11 = axs1[2,0] # lognormal rate - varSb
    ax12 = axs1[3,0] # residue rate - varSb
    ax13 = axs1[0,1] # lognormal rate - S2
    ax14 = axs1[1,1] # residue rate - S2
    ax15 = axs1[2,1] # lognormal rate - SDNR
    ax16 = axs1[3,1] # residue rate - SDNR


    #parameters of grid
    y_begin=0.1
    x_begin=0.12
    height_res=0.1
    height_dat=0.22
    space_s=0.01
    space_l=0.065
    central_space=0.1
    h_space=0.5
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

    

    ##########grid plot lognormal model##############

    
    ax9.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax10.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax11.set_position([x_begin, y_begin+height_res+space_s, width_g, height_dat])
    ax12.set_position([x_begin, y_begin, width_g, height_res])
    ax13.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax14.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax15.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s, width_g, height_dat])
    ax16.set_position([x_begin+width_g+central_space, y_begin, width_g, height_res])



    #################################### PLOT DISCRETE MODEL #################################################

    plt.figure(1)
    #ax1.set_title("Discrete rate model", fontsize=legend_fs)
    ax1.fill_between(discr['T'], discr['Sb_av']-discr['Sb_std'], discr['Sb_av']+discr['Sb_std'], color="red", alpha=0.2)
    ax1.plot(discr['T'], discr['Sb_av'], "-", color="blue", label="Simulation")
    ax1.plot(discr['T'], th_discr['Sb_th'], "--", color="red", label="Theory")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    #ax1.legend(title=r"$\langle S_b \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    ax1.legend(fontsize=legend_fs, framealpha=0.75)
    ax1.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_xscale('log')
    ax1.set_xticklabels([])
    #ax1.grid()
    
    #ax1.legend(title=r"$\langle S_b \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)

    #ax9.set_title("Residue", fontsize=legend_fs)
    #ax9.fill_between(discr['T'], discr['Sb_av']-discr['Sb_std'], discr['Sb_av']+discr['Sb_std'], color="red", alpha=0.2)
    ax2.plot(discr['T'], abs(discr['Sb_av']-th_discr['Sb_th'])/th_discr['Sb_th']*100, "-", color="green", label="Simulation")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax2.set_ylabel(r"$\dfrac{\langle S_b \rangle - \langle S_{b} \rangle ^{th}}{\langle S_{b} \rangle ^{th}}$ ", fontsize=tick_fs)
    ax2.set_xlabel('T training patterns', fontsize=tick_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.set_xscale('log')
    #ax2.grid()

   

    #ax3.set_title("Discrete rate model", fontsize=legend_fs)
    #ax3.fill_between(discr['T'], discr['varSb_av']-discr['varSb_std'], discr['varSb_av']+discr['varSb_std'], color="blue", alpha=0.2)
    ax3.plot(discr['T'], discr['varSb_av'], "-", color="blue", label="Simulation")
    ax3.plot(discr['T'], th_discr['varSb_th'], "--", color="red", label="Theory")
    #ax3.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    ax3.legend(fontsize=legend_fs, framealpha=0.75)
    #ax3.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax3.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax3.tick_params(labelsize=tick_fs)
    ax3.set_xscale('log')
    #ax3.grid()
    ax3.set_xticklabels([])
    #ax3.legend(fontsize=legend_fs, framealpha=0.75)

    #ax3.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    ax3.legend(fontsize=legend_fs, framealpha=0.75)


    ax4.plot(discr['T'], abs(discr['varSb_av']-th_discr['varSb_th'])/th_discr['varSb_th']*100, "-", color="green", label="Simulation")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax4.set_ylabel(r"$\dfrac{\sigma^2_b - \sigma^2_{b^{th}})}{\sigma^2_{b^{th}}}$ ", fontsize=tick_fs)
    ax4.set_xlabel('T training patterns', fontsize=tick_fs)
    ax4.tick_params(labelsize=tick_fs)
    ax4.set_xscale('log')
    #ax4.grid()

   

   


    #ax5.fill_between(discr['T'], discr['S2_av']-discr['S2_std'], discr['S2_av']+discr['S2_std'], color="blue", alpha=0.2)
    ax5.plot(discr['T'], discr['S2_av'], "-", color="blue", label="Simulation")
    ax5.plot(discr['T'], th_discr['S2_th'], "--", color="red", label="Theory")
    #ax5.legend(title=r"$\langle S_c \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    ax5.legend(fontsize=legend_fs, framealpha=0.75)
    #ax5.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax5.set_ylabel(r"$\langle S_c \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax5.tick_params(labelsize=tick_fs)
    ax5.set_xscale('log')
    #ax5.grid()
    #ax5.legend(fontsize=legend_fs, framealpha=0.75)
    #ax5.legend(title=r"$\langle S_c \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    ax5.set_xticklabels([])


    ax6.plot(discr['T'], abs(discr['S2_av']-th_discr['S2_th'])/th_discr['S2_th']*100, "-", color="green", label="Simulation")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax6.set_ylabel(r"$\dfrac{\langle S_c \rangle - \langle S_{c} \rangle ^{th}}{\langle S_{c} \rangle ^{th}}$ ", fontsize=tick_fs)
    ax6.set_xlabel('T training patterns', fontsize=tick_fs)
    ax6.tick_params(labelsize=tick_fs)
    ax6.set_xscale('log')
    #ax6.grid()
    
    
    
    
    ax7.plot(discr['T'], np.abs(discr['S2_av']-discr['Sb_av'])/np.sqrt(discr['varSb_av']), "-", color="blue", label="Simulation")
    ax7.plot(discr['T'], np.abs(th_discr['S2_th']-th_discr['Sb_th'])/np.sqrt(th_discr['varSb_th']), "--", color="red", label="Theory")
    #ax7.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax7.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax7.tick_params(labelsize=tick_fs)
    ax7.set_xscale('log')
    #ax7.grid()
    #ax7.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    ax7.legend(fontsize=legend_fs, framealpha=0.75)
    ax7.set_xticklabels([])


    ax8.plot(discr['T'], abs(np.abs(discr['S2_av']-discr['Sb_av'])/np.sqrt(discr['varSb_av'])- np.abs(th_discr['S2_th']-th_discr['Sb_th'])/np.sqrt(th_discr['varSb_th']))/ (np.abs(th_discr['S2_th']-th_discr['Sb_th'])/np.sqrt(th_discr['varSb_th']))*100, "-", color="green", label="Simulation")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax8.set_ylabel(r"$\dfrac{SDNR - SDNR^{th}}{SDNR^{th}}$ ", fontsize=tick_fs)
    ax8.set_xlabel('T training patterns', fontsize=tick_fs)
    ax8.tick_params(labelsize=tick_fs)
    #ax8.set_xticklabels([])
    ax8.set_xscale('log')
    #ax8.grid()
    #ax8.legend(fontsize=legend_fs, framealpha=0.75)

   
    #fig1.subplots_adjust(wspace=0.6)
    fig.subplots_adjust(bottom=0.1, top=0.97, right=0.975, left=0.1)
    #fig.suptitle('Discrete Model', fontsize=25)
    fig.savefig('discrete.png')
    fig.show()







    ############################## PLOT LOGNORMAL MODEL #########################################################
    
    colors = ["red", "blue", "black", "green", "magenta", "darkkhaki"]
    th_colors = ["orange", "cornflowerblue", "grey", "limegreen", "plum", "khaki"]

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    plt.figure(2)
    #ax9.set_title("Lognormal rate model", fontsize=legend_fs)
    #ax9.fill_between(ln['T'], ln['Sb_av']-ln['Sb_std'], ln['Sb_av']+ln['Sb_std'], color="blue", alpha=0.2)
    #ax9.plot(ln['T'], ln['Sb_av'], "-", color="blue", label="Sim - w/out noise")
    #ax9.plot(ln['T'], th_ln['Sb_th'], "--", color="cornflowerblue", label="Th - w/out noise")
    #ax9.fill_between(ln_noise['T'], ln_noise['Sb_av']-ln_noise['Sb_std'], ln_noise['Sb_av']+ln_noise['Sb_std'], color="cornflowerblue", alpha=0.2)
        
    l1_th, = ax9.plot(th_ln_noise[0]['T'], th_ln_noise[0]['Sb_th'], "--", color=th_colors[0], label="Th - " + label[0] + " noise")
    l1, = ax9.plot(ln_noise[0]['T'], ln_noise[0]['Sb_av'], "^", color=colors[0], label="Sim - " + label[0] + " noise")

    l2_th, = ax9.plot(th_ln_noise[1]['T'], th_ln_noise[1]['Sb_th'], "--", color=th_colors[1], label="Th - " + label[1] + " noise")
    l2, = ax9.plot(ln_noise[1]['T'], ln_noise[1]['Sb_av'], "^", color=colors[1], label="Sim - " + label[1] + " noise")

    l3_th, = ax9.plot(th_ln_noise[2]['T'], th_ln_noise[2]['Sb_th'], "--", color=th_colors[2], label="Th - " + label[2] + " noise")
    l3, = ax9.plot(ln_noise[2]['T'], ln_noise[2]['Sb_av'], "^", color=colors[2], label="Sim - " + label[2] + " noise")

    l4_th, = ax9.plot(th_ln_noise[3]['T'], th_ln_noise[3]['Sb_th'], "--", color=th_colors[3], label="Th - " + label[3] + " noise")
    l4, = ax9.plot(ln_noise[3]['T'], ln_noise[3]['Sb_av'], "^", color=colors[3], label="Sim - " + label[3] + " noise")

    l5_th, = ax9.plot(th_ln_noise[4]['T'], th_ln_noise[4]['Sb_th'], "--", color=th_colors[4], label="Th - " + label[4] + " noise")
    l5, = ax9.plot(ln_noise[4]['T'], ln_noise[4]['Sb_av'], "^", color=colors[4], label="Sim - " + label[4] + " noise")
    
    l6_th, = ax9.plot(th_ln_noise[5]['T'], th_ln_noise[5]['Sb_th'], "--", color=th_colors[5], label="Th - " + label[5] + " noise")
    l6, = ax9.plot(ln_noise[5]['T'], ln_noise[5]['Sb_av'], "^", color=colors[5], label="Sim - " + label[5] + " noise")
    
    ax9.text(-0.1, 1.05, "A", weight="bold", fontsize=30, color='k', transform=ax9.transAxes)
    #ax9.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax9.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax9.tick_params(labelsize=tick_fs)
    #ax9.set_xscale('log')
    ax9.set_xticklabels([])
    #ax9.grid()
    #ax9.legend(title=r"$\langle S_b \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    #ax9.legend(fontsize=legend_fs, framealpha=0.75)
    ax9.legend(handles =[(plt.plot([], marker='s', color=l1.get_color())[0],plt.plot([], marker='s', color=l1_th.get_color())[0]),
                (plt.plot([], marker='s', color=l2.get_color())[0],plt.plot([], marker='s', color=l2_th.get_color())[0]), 
                (plt.plot([], marker='s', color=l3.get_color())[0],plt.plot([], marker='s', color=l3_th.get_color())[0]), 
                (plt.plot([], marker='s', color=l4.get_color())[0],plt.plot([], marker='s', color=l4_th.get_color())[0]), 
                (plt.plot([], marker='s', color=l5.get_color())[0],plt.plot([], marker='s', color=l5_th.get_color())[0]), 
                (plt.plot([], marker='s', color=l6.get_color())[0],plt.plot([], marker='s', color=l6_th.get_color())[0])], 
                labels = ['Sim/Th no noise', 'Sim/Th 1 Hz noise', 'Sim/Th 2 Hz noise', 'Sim/Th 3 Hz noise', 'Sim/Th 4 Hz noise', 'Sim/Th 5 Hz noise'], numpoints=6,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_fs, framealpha=0.75)



    #ax10.set_title("Residue", fontsize=legend_fs)
    #ax9.fill_between(discr['T'], discr['Sb_av']-discr['Sb_std'], discr['Sb_av']+discr['Sb_std'], color="red", alpha=0.2)
    
    l1, = ax10.plot(ln_noise[0]['T'], abs(ln_noise[0]['Sb_av']-th_ln_noise[0]['Sb_th'])/th_ln_noise[0]['Sb_th']*100, "--", color=colors[0], label=label[0] + " noise")
    l2, = ax10.plot(ln_noise[1]['T'], abs(ln_noise[1]['Sb_av']-th_ln_noise[1]['Sb_th'])/th_ln_noise[1]['Sb_th']*100, "--", color=colors[1], label=label[1] + " noise")
    l3, = ax10.plot(ln_noise[2]['T'], abs(ln_noise[2]['Sb_av']-th_ln_noise[2]['Sb_th'])/th_ln_noise[2]['Sb_th']*100, "--", color=colors[2], label=label[2] + " noise")
    l4, = ax10.plot(ln_noise[3]['T'], abs(ln_noise[3]['Sb_av']-th_ln_noise[3]['Sb_th'])/th_ln_noise[3]['Sb_th']*100, "--", color=colors[3], label=label[3] + " noise")
    l5, = ax10.plot(ln_noise[4]['T'], abs(ln_noise[4]['Sb_av']-th_ln_noise[4]['Sb_th'])/th_ln_noise[4]['Sb_th']*100, "--", color=colors[4], label=label[4] + " noise")
    l6, = ax10.plot(ln_noise[5]['T'], abs(ln_noise[4]['Sb_av']-th_ln_noise[5]['Sb_th'])/th_ln_noise[5]['Sb_th']*100, "--", color=colors[5], label=label[5] + " noise")
    
    ax10.set_ylabel(r"$\dfrac{\langle S_b \rangle - \langle S_{b} \rangle ^{th}}{\langle S_{b} \rangle ^{th}}\quad$[%] ", fontsize=tick_fs)
    ax10.set_xlabel('T training patterns', fontsize=tick_fs)
    ax10.tick_params(labelsize=tick_fs)
    #ax10.set_xscale('log')
    if(sat==False):
        ax10.set_ylim(0, 0.06)
    ax10.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax10.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    #ax10.grid()
    ax10.legend(fontsize=legend_fs, framealpha=0.75, ncol=2)


    
    #ax4.fill_between(ln_noise['T'], ln_noise['varSb_av']-ln_noise['varSb_std'], ln_noise['varSb_av']+ln_noise['varSb_std'], color="cornflowerblue", alpha=0.2)
    #ax4.set_xlabel("T (training patterns)", fontsize=tick_fs)
    #ax4.tick_params(labelsize=tick_fs)
    #ax4.set_xscale('log')
    #ax4.grid()

    #ax11.set_title("Lognormal rate model", fontsize=legend_fs)
    #ax4.fill_between(ln['T'], ln['varSb_av']-ln['varSb_std'], ln['varSb_av']+ln['varSb_std'], color="blue", alpha=0.2)
    #ax11.plot(ln['T'], ln['varSb_av'], "-", color="blue", label="Sim - w/out noise")
    #ax11.plot(ln['T'], th_ln['varSb_th'], "--", color="cornflowerblue", label="Th - w/out noise")
    ax11.text(-0.1, 1.05, "C", weight="bold", fontsize=30, color='k', transform=ax11.transAxes)
    for i in range(len(ln_noise)):
        ax11.plot(ln_noise[i]['T'], th_ln_noise[i]['varSb_th'], "--", color=th_colors[i], label="Th - " + label[i] + " noise")
        ax11.plot(ln_noise[i]['T'], ln_noise[i]['varSb_av'], "^", color=colors[i], label="Sim - " + label[i] + " noise")
    ax11.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    #ax11.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]", fontsize=tick_fs)
    ax11.tick_params(labelsize=tick_fs)
    #ax11.set_xscale('log')
    ax11.set_xticklabels([])
    #ax11.grid()
    #ax11.legend(title=r"$\sigma^2_b$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)

    #ax11.legend(fontsize=legend_fs, framealpha=0.75)

    
    #ax12.plot(discr['T'], abs(ln['varSb_av']-th_ln['varSb_th'])/th_ln['varSb_th']*100, "-", color="blue", label="w/out noise")
    for i in range(len(ln_noise)):
        ax12.plot(ln_noise[i]['T'], abs(ln_noise[i]['varSb_av']-th_ln_noise[i]['varSb_th'])/th_ln_noise[i]['varSb_th']*100, "--", color=colors[i], label=label[i] + " noise")    
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    #ax12.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax12.set_ylabel(r"$\dfrac{\sigma^2_b - \sigma^2_{b^{th}}}{\sigma^2_{b^{th}}}\quad$[%]", fontsize=tick_fs)
    ax12.set_xlabel('T training patterns', fontsize=tick_fs)
    ax12.tick_params(labelsize=tick_fs)
    ax12.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax12.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    #ax12.set_xscale('log')
    #ax12.set_xticklabels([])
    #ax12.grid()
    #ax12.legend(fontsize=legend_fs, framealpha=0.75)



    #ax6.fill_between(ln['T'], ln['S2_av']-ln['S2_std'], ln['S2_av']+ln['S2_std'], color="blue", alpha=0.2)
    #ax13.plot(ln['T'], ln['S2_av'], "-", color="blue", label="Sim - w/out noise")
    #ax13.plot(ln['T'], th_ln['S2_th'], "--", color="cornflowerblue", label="Th - w/out noise")


    #ax6.fill_between(ln_noise['T'], ln_noise['S2_av']-ln_noise['S2_std'], ln_noise['S2_av']+ln_noise['S2_std'], color="cornflowerblue", alpha=0.2)
    ax13.text(-0.1, 1.05, "B", weight="bold", fontsize=30, color='k', transform=ax13.transAxes)
    for i in range(len(ln_noise)):
        ax13.plot(ln_noise[i]['T'], th_ln_noise[i]['S2_th'], "--", color=th_colors[i], label="Th - " + label[i] + " noise")
        ax13.plot(ln_noise[i]['T'], ln_noise[i]['S2_av'], "^", color=colors[i], label="Sim - " + label[i] + " noise")
        
    #ax6.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax13.set_ylabel(r"$\langle S_c \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax13.tick_params(labelsize=tick_fs)
    #ax13.set_xscale('log')
    ax13.set_xticklabels([])
    #ax13.grid()
    #ax13.legend(title=r" $\langle S_c \rangle$", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    #ax13.legend(fontsize=legend_fs, framealpha=0.75)

   # plt.subplots_adjust(hspace=0.1)
    
    #ax14.plot(discr['T'], abs(ln['S2_av']-th_ln['S2_th'])/th_ln['S2_th']*100, "-", color="blue", label="w/out noise")
    for i in range(len(ln_noise)):
        ax14.plot(ln_noise[i]['T'], abs(ln_noise[i]['S2_av']-th_ln_noise[i]['S2_th'])/th_ln_noise[i]['S2_th']*100, "--", color=colors[i], label=label[i] + " noise")
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax14.set_ylabel(r"$\dfrac{\langle S_{c} \rangle - \langle S_{c}\rangle ^{th}}{\langle S_{c}\rangle ^{th}}\quad$[%]", fontsize=tick_fs)
    ax14.set_xlabel('T training patterns', fontsize=tick_fs)
    ax14.tick_params(labelsize=tick_fs)
    ax14.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax14.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    #ax14.set_xscale('log')
    #ax14.set_xticklabels([])
    #ax14.grid()
    #ax14.legend(fontsize=legend_fs, framealpha=0.75)


   # plt.subplots_adjust(hspace=0.5)


    
    #ax15.plot(ln['T'], np.abs(ln['S2_av']-ln['Sb_av'])/np.sqrt(ln['varSb_av']), "-", color="blue", label="Sim - w/out noise")
    #ax15.plot(ln['T'], np.abs(th_ln['S2_th']-th_ln['Sb_th'])/np.sqrt(th_ln['varSb_th']), "--", color="cornflowerblue", label="Th - w/out noise")
    for i in range(len(ln_noise)):
        ax15.plot(ln_noise[i]['T'], np.abs(th_ln_noise[i]['S2_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']), "--", color=th_colors[i], label='_nolegend_')
        ax15.plot(ln_noise[i]['T'], np.abs(ln_noise[i]['S2_av']-ln_noise[i]['Sb_av'])/np.sqrt(ln_noise[i]['varSb_av']), "^", color=colors[i], label='_nolegend_')
    
    #ax15.axhline(y=SDNR_thr, color='r', linestyle='-', label='SDNR threshold')
    ax15.plot(np.linspace(5000,100000,5),SDNR_thr*np.ones(5), linestyle='-', label='SDNR threshold')

  #  plt.subplots_adjust(hspace=0.1)


    #ax15.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax15.text(-0.1, 1.05, "D", weight="bold", fontsize=30, color='k', transform=ax15.transAxes)
    ax15.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax15.tick_params(labelsize=tick_fs)
    ax15.set_xscale('log')
    ax15.set_xticklabels([])
    ax15.legend(fontsize=legend_fs)
    #ax15.grid()
    #ax15.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    #ax15.legend(fontsize=legend_fs, framealpha=0.75)

    #ax16.plot(discr['T'], abs(np.abs(ln['S2_av']-ln['Sb_av'])/np.sqrt(ln['varSb_av'])-np.abs(th_ln['S2_th']-th_ln['Sb_th'])/np.sqrt(th_ln['varSb_th']))/(np.abs(th_ln['S2_th']-th_ln['Sb_th'])/np.sqrt(th_ln['varSb_th']))*100, "-", color="blue", label="w/out noise")
    for i in range(len(ln_noise)):
        ax16.plot(ln_noise[i]['T'], abs(np.abs(ln_noise[i]['S2_av']-ln_noise[i]['Sb_av'])/np.sqrt(ln_noise[i]['varSb_av'])-np.abs(th_ln_noise[i]['S2_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']))/(np.abs(th_ln_noise[i]['S2_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']))*100, "--", color=colors[i], label=label[i] + " noise")
    
    #ax1.set_xlabel("T (training patterns)", fontsize=tick_fs)
    ax16.set_ylabel(r"$\dfrac{SDNR - SDNR^{th}}{SDNR^{th}}\quad$[%] ", fontsize=tick_fs)
    ax16.set_xlabel('T training patterns', fontsize=tick_fs)
    ax16.tick_params(labelsize=tick_fs)
    ax16.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax16.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    #ax16.set_xscale('log')
    #ax16.set_yscale('log')
    #ax16.set_xticklabels([])
    #ax16.grid()
    #ax16.legend(title=r"SDNR", fontsize=legend_fs, title_fontsize=legend_fs, framealpha=0.75)
    #ax16.legend(fontsize=legend_fs, framealpha=0.75)

   
    #fig1.subplots_adjust(wspace=0.6)
    #fig1.subplots_adjust(bottom=0.1, top=0.97, right=0.975, left=0.1)
    #fig1.suptitle('Lognormal Model', fontsize=25)
    #fig1.savefig('lognormal.png')
    fig1.savefig('lognormal' + saturation +'.png')
    #fig1.show()
    
    #plt.savefig("structural_plasticity.png")

    
saturation = True
if saturation==True:
    sat = "_saturation"
else:
    sat = ""

# values extracted from simulations
discr_rate = get_data("../simulations/discr_rate_simulations", 10)
ln_rate = get_data("../simulations/no_noise_simulations", 10)
ln_rate_noise1 = get_data("../simulations/noise_1Hz_simulations"+sat, 10)
ln_rate_noise2 = get_data("../simulations/noise_2Hz_simulations"+sat, 10)
ln_rate_noise3 = get_data("../simulations/noise_3Hz_simulations"+sat, 10)
ln_rate_noise4 = get_data("../simulations/noise_4Hz_simulations"+sat, 10)
ln_rate_noise5 = get_data("../simulations/noise_5Hz_simulations"+sat, 10)


# theoretical values
th_discr = get_th_data("../simulations/discr_rate_simulations")
th_ln = get_th_data("../simulations/no_noise_simulations")
th_ln_noise1 = get_th_data("../simulations/noise_1Hz_simulations"+sat)
th_ln_noise2 = get_th_data("../simulations/noise_2Hz_simulations"+sat)
th_ln_noise3 = get_th_data("../simulations/noise_3Hz_simulations"+sat)
th_ln_noise4 = get_th_data("../simulations/noise_4Hz_simulations"+sat)
th_ln_noise5 = get_th_data("../simulations/noise_5Hz_simulations"+sat)

prob_thr=0.9
ln_rate_noise = [ln_rate, ln_rate_noise1, ln_rate_noise2, ln_rate_noise3, ln_rate_noise4, ln_rate_noise5]
th_ln_noise = [th_ln, th_ln_noise1, th_ln_noise2, th_ln_noise3, th_ln_noise4, th_ln_noise5]
label = ["No", "1 Hz", "2 Hz", "3 Hz", "4 Hz", "5 Hz"]


plot_data(discr_rate, th_discr, ln_rate, th_ln, ln_rate_noise, th_ln_noise, label, sat,prob_thr)

plt.show()
