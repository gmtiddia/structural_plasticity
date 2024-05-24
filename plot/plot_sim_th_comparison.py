import os
from scipy.optimize import curve_fit
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
    beta1 = 1.0 - params['alpha1']
    beta2 = 1.0 - params['alpha2']
    # average rate layer 1
    nu_av_1 = params['alpha1']*params['nu_h_1'] + beta1*params['nu_l_1']
    # average rate layer 2
    nu_av_2 = params['alpha2']*params['nu_h_2'] + beta2*params['nu_l_2']

    # rate threshold for both layers
    if(params['lognormal_rate']==1):
        sigma_ln1 = erfm1(beta1) - erfm1(beta1*params['nu_l_1']/nu_av_1)
        mu_ln1 = math.log(nu_av_1) - sigma_ln1*sigma_ln1/2.0
        yt_ln1 = erfm1(beta1)*sigma_ln1 + mu_ln1
        rt1 = np.exp(yt_ln1)
        sigma_ln2 = erfm1(beta2) - erfm1(beta2*params['nu_l_2']/nu_av_2)
        mu_ln2 = math.log(nu_av_2) - sigma_ln2*sigma_ln2/2.0
        yt_ln2 = erfm1(beta2)*sigma_ln2 + mu_ln2
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
        var_r1 = rsq1 - nu_av_1*nu_av_1
    
    # variance of k
    k2 = params['C']*(params['C'] - 1.0)*((1.0 - (2.0 - params['alpha1'])*params['alpha1']*params['alpha2'])**T) - params['C']*(2.0*params['C'] - 1.0)*((1.0 - params['alpha1']*params['alpha2'])**T) + params['C']*params['C']
    var_k = k2 - k*k

    # theoretical value of Sb
    Sb = params['Ws']*k*nu_av_1 + params['Wb']*(params['C']-k)*nu_av_1

    # theoretical value of Sc
    if(params['r']==0):
        # without rewiring
        Sc = params['nu_h_1']*params['Ws']*params['alpha1']*params['C'] + params['nu_l_1']*(1.0-params['alpha1'])*(params['Wb']*params['C'] + (params['Ws'] - params['Wb'])*k)
    else:
        # with rewiring
        r = params['r']
        b = (1.-(1.-params['alpha1']*params['alpha2'])**(T+r))/(1.-(1.-params['alpha1']*params['alpha2'])**r)
        av_pt = 1. - (b*r)/(T+r)
        av_kt_first = av_pt*params['C']*(1.0-params['alpha1'])
        # w/ rewiring
        Sc = params['nu_h_1']*params['Ws']*params['alpha1']*params['C'] + nu_av_1*(1.0-params['alpha1'])*(params['Wb']*params['C'] + (params['Ws'] - params['Wb'])*k) - params['Ws']*(nu_av_1 - params['nu_l_1'])*av_kt_first

    # if C is constant
    if(params['connection_rule']==0):
        # variance of Sb for C fixed
        varSb = (params['Ws']*params['Ws']*k + params['Wb']*params['Wb']*(params['C']-k))*var_r1 + (params['Ws'] - params['Wb'])*(params['Ws'] - params['Wb'])*nu_av_1*nu_av_1*var_k
    # if C is driven from Poisson distribution
    else:
        # average and variance of the Poisson distribution is C
        C_m = params['C']
        var_C = params['C']
        C2_m = params['C']*(params['C'] + 1.0)
        eta = (1.0 - params['alpha1']*params['alpha2'])**T
        csi = (1.0 - (2.0 - params['alpha1'])*params['alpha1']*params['alpha2'])**T
        # variance of Sb for C variable
        varSb = ((params['Wb'] + p*(params['Ws'] - params['Wb']))**2.0)*nu_av_1*nu_av_1*var_C + C_m*(p*params['Ws']*params['Ws'] + eta*params['Wb']*params['Wb'])*var_r1 + ((params['Ws'] - params['Wb'])**2)*nu_av_1*nu_av_1*((C2_m - C_m)*csi + C_m*eta - C2_m*eta*eta)

    # if we add noise on the test patterns
    if(params['noise_flag']==1):
        beta = params['max_noise_dev']
        Z = Phi(beta) - Phi(-beta)
        var_noise = params['rate_noise']*params['rate_noise']*(1.0 - 2.0*beta*phi1(beta)/Z)
        var_S_noise = (params['Ws']*params['Ws']*k + params['Wb']*params['Wb']*(params['C'] - k))*var_noise
        # add noise contribution to the variance of Sb
        #print("Old varSb: {}".format(varSb))
        varSb += var_S_noise
        #print("New varSb: {}".format(varSb))

    return({"T": T, "Sb": Sb, "varSb": varSb, "Sc": Sc})


def get_th_data(path):
    """
    Returns a DataFrame with the theoretical estimation of Sb, Sc and Var(Sb) given the simulation parameters.

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
        dum_Sc = []

        for i in range(len(T)):
            dum_dict = get_th_set(path+"/5000", T[i])
            dum_Sb.append(dum_dict['Sb'])
            dum_varSb.append(dum_dict['varSb'])
            dum_Sc.append(dum_dict['Sc'])

        dic = {"T": T, "Sb_th": dum_Sb, "varSb_th": dum_varSb, "Sc_th": dum_Sc}

        data = pd.DataFrame(dic)
        data.to_csv(path+"/th_values.csv", index=False)
        print("csv generated")
        return(data)


def get_data(path, seeds):
    """
    Returns a dataframe with averaged values of Sb, Sc and Var(Sb) over the seeds.
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
        Sc_av = np.zeros(len(T)); Sc_std = np.zeros(len(T))
        Sb_av = np.zeros(len(T)); Sb_std = np.zeros(len(T))
        varSb_av = np.zeros(len(T)); varSb_std = np.zeros(len(T))

        # extract values from data
        for t, dir in enumerate(subfolders):
            Sb_dum = []
            Sc_dum = []
            varSb_dum = []
            for i in range(seeds):
                mem_out = np.loadtxt(path+"/"+str(T[t])+"/mem_out_000"+str(i)+"_0000.dat")
                Sb_dum.append(np.average(mem_out[:,1]))
                Sc_dum.append(np.average(mem_out[:,2]))
                varSb_dum.append(np.average(mem_out[:,3]))
            Sc_av[t]=np.average(Sc_dum)
            Sb_av[t]=np.average(Sb_dum)
            varSb_av[t]=np.average(varSb_dum)
            Sb_std[t]=np.std(Sb_dum)
            Sc_std[t]=np.std(Sc_dum)
            varSb_std[t]=np.std(varSb_dum)
        
        # free some memory
        del mem_out, Sb_dum, Sc_dum, varSb_dum

        # save values on a DataFrame
        data = {"T": T, "Sb_av": Sb_av, "Sb_std": Sb_std, "varSb_av": varSb_av, "varSb_std": varSb_std, "Sc_av": Sc_av, "Sc_std": Sc_std}
        data = pd.DataFrame(data)
        #print(data)

        data.to_csv(path+"/values.csv", index=False)
        print("csv generated")
        return(data)


def plot_data(ln_noise, th_ln_noise, label, saturation, prob_thr):

    def SDNR_thres(prob_thr):
        
        SDNR_thr=erfinv(2*prob_thr-1) *2*np.sqrt(2)
        return(SDNR_thr)
    
    SDNR_thr=SDNR_thres(prob_thr)
    print('Having P_C=' + str(prob_thr)+ ' the SDNR threshold is ' + str(SDNR_thr))

    """
    Plot data and saves the plot

    Parameters
    ----------
    ln, th_ln: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and Sc for lognormal rate model
    ln_noise, th_ln_noise: list of pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and Sc for lognormal rate model with noise

    """

    # fontsize params
    legend_fs =20
    tick_fs = 25

    #width and height ratios of subplots
    widths= [1, 1, 1, 1]
    heights= [3, 1, 3, 1]

    fig1, axs1 = plt.subplots(ncols=2, nrows=4, figsize = (20,18),constrained_layout=True,gridspec_kw={'height_ratios': heights})
  

    #subplots
    ax1 = axs1[0,0] # lognormal rate - Sb
    ax2 = axs1[1,0] # residue - Sb
    ax3 = axs1[2,0] # lognormal rate - varSb
    ax4 = axs1[3,0] # residue rate - varSb
    ax5 = axs1[0,1] # lognormal rate - Sc
    ax6 = axs1[1,1] # residue rate - Sc
    ax7 = axs1[2,1] # lognormal rate - SDNR
    ax8 = axs1[3,1] # residue rate - SDNR


    #parameters of grid
    y_begin=0.22
    x_begin=0.12
    height_res=0.1
    height_dat=0.22
    space_s=0.01
    space_l=0.065
    central_space=0.1
    h_space=0.5
    width_g=0.35


    ##########grid plot lognormal model##############

    
    ax1.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax2.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax3.set_position([x_begin, y_begin+height_res+space_s, width_g, height_dat])
    ax4.set_position([x_begin, y_begin, width_g, height_res])
    ax5.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax6.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax7.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s, width_g, height_dat])
    ax8.set_position([x_begin+width_g+central_space, y_begin, width_g, height_res])




    ############################## PLOT LOGNORMAL MODEL #########################################################
    
    colors = ["magenta", "darkkhaki", "red", "blue", "black", "green", "darkcyan", "sienna"]
    th_colors = ["plum", "khaki", "orange", "cornflowerblue", "grey", "limegreen", "cyan", "sandybrown"]

    from matplotlib.legend_handler import HandlerTuple
        
    l1_th, = ax1.plot(th_ln_noise[0]['T'], th_ln_noise[0]['Sb_th'], "--", color=th_colors[0], label="Th - " + label[0] + " noise")
    l1, = ax1.plot(ln_noise[0]['T'], ln_noise[0]['Sb_av'], "^", markersize=7, color=colors[0], label="Sim - " + label[0] + " noise")

    l2_th, = ax1.plot(th_ln_noise[1]['T'], th_ln_noise[1]['Sb_th'], "--", color=th_colors[1], label="Th - " + label[1] + " noise")
    l2, = ax1.plot(ln_noise[1]['T'], ln_noise[1]['Sb_av'], "^", markersize=7, color=colors[1], label="Sim - " + label[1] + " noise")

    l3_th, = ax1.plot(th_ln_noise[2]['T'], th_ln_noise[2]['Sb_th'], "--", color=th_colors[2], label="Th - " + label[2] + " noise")
    l3, = ax1.plot(ln_noise[2]['T'], ln_noise[2]['Sb_av'], "^", markersize=7, color=colors[2], label="Sim - " + label[2] + " noise")

    l4_th, = ax1.plot(th_ln_noise[3]['T'], th_ln_noise[3]['Sb_th'], "--", color=th_colors[3], label="Th - " + label[3] + " noise")
    l4, = ax1.plot(ln_noise[3]['T'], ln_noise[3]['Sb_av'], "^", markersize=7, color=colors[3], label="Sim - " + label[3] + " noise")

    l5_th, = ax1.plot(th_ln_noise[4]['T'], th_ln_noise[4]['Sb_th'], "--", color=th_colors[4], label="Th - " + label[4] + " noise")
    l5, = ax1.plot(ln_noise[4]['T'], ln_noise[4]['Sb_av'], "^", markersize=7, color=colors[4], label="Sim - " + label[4] + " noise")
    
    l6_th, = ax1.plot(th_ln_noise[5]['T'], th_ln_noise[5]['Sb_th'], "--", color=th_colors[5], label="Th - " + label[5] + " noise")
    l6, = ax1.plot(ln_noise[5]['T'], ln_noise[5]['Sb_av'], "^", markersize=7, color=colors[5], label="Sim - " + label[5] + " noise")

    l7_th, = ax1.plot(th_ln_noise[6]['T'], th_ln_noise[6]['Sb_th'], "--", color=th_colors[6], label="Th - " + label[6] + " noise")
    l7, = ax1.plot(ln_noise[6]['T'], ln_noise[6]['Sb_av'], "^", markersize=7, color=colors[6], label="Sim - " + label[6] + " noise")

    l8_th, = ax1.plot(th_ln_noise[7]['T'], th_ln_noise[7]['Sb_th'], "--", color=th_colors[7], label="Th - " + label[7] + " noise")
    l8, = ax1.plot(ln_noise[7]['T'], ln_noise[7]['Sb_av'], "^", markersize=7, color=colors[7], label="Sim - " + label[7] + " noise")
    
    ax1.text(-0.1, 1.05, "A", weight="bold", fontsize=30, color='k', transform=ax1.transAxes)
    ax1.set_ylabel(r"$\langle S_b \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax1.set_xticklabels([])
    ax8.legend(handles =[(plt.scatter([], [], marker='^', s=100, color=l1.get_color()),plt.plot([], linestyle='--', lw=2, color=l1_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l1_th.get_color())[0]),
                (plt.scatter([], [], marker='^', s=100, color=l2.get_color()),plt.plot([], linestyle='--', lw=2, color=l2_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l2_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l3.get_color()),plt.plot([], linestyle='--', lw=2, color=l3_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l3_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l4.get_color()),plt.plot([], linestyle='--', lw=2, color=l4_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l4_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l5.get_color()),plt.plot([], linestyle='--', lw=2, color=l5_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l5_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l6.get_color()),plt.plot([], linestyle='--', lw=2, color=l6_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l6_th.get_color())[0]),
                (plt.scatter([], [], marker='^', s=100, color=l7.get_color()),plt.plot([], linestyle='--', lw=2, color=l7_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l7_th.get_color())[0]),
                (plt.scatter([], [], marker='^', s=100, color=l8.get_color()),plt.plot([], linestyle='--', lw=2, color=l8_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l8_th.get_color())[0])], 
                
                labels = ['No noise', '0.2 Hz std noise', '0.4 Hz std noise', '0.6 Hz std noise', '0.8 Hz std noise', '1 Hz std noise', '1.5 Hz std noise', '2 Hz std noise'], numpoints=8, ncol=2,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_fs+2, framealpha=1.0, handlelength=5, title=r"Simulation ($\blacktriangle$), theoretical estimation (--) and comparison ($\cdots$)", title_fontsize=legend_fs+5,
              bbox_to_anchor=(0.63, -1.5),loc='center right')

    
    l1, = ax2.plot(ln_noise[0]['T'], abs(ln_noise[0]['Sb_av']-th_ln_noise[0]['Sb_th'])/th_ln_noise[0]['Sb_th']*100, ":", lw=2, color=colors[0], label=label[0] + " noise")
    l2, = ax2.plot(ln_noise[1]['T'], abs(ln_noise[1]['Sb_av']-th_ln_noise[1]['Sb_th'])/th_ln_noise[1]['Sb_th']*100, ":", lw=2, color=colors[1], label=label[1] + " noise")
    l3, = ax2.plot(ln_noise[2]['T'], abs(ln_noise[2]['Sb_av']-th_ln_noise[2]['Sb_th'])/th_ln_noise[2]['Sb_th']*100, ":", lw=2, color=colors[2], label=label[2] + " noise")
    l4, = ax2.plot(ln_noise[3]['T'], abs(ln_noise[3]['Sb_av']-th_ln_noise[3]['Sb_th'])/th_ln_noise[3]['Sb_th']*100, ":", lw=2, color=colors[3], label=label[3] + " noise")
    l5, = ax2.plot(ln_noise[4]['T'], abs(ln_noise[4]['Sb_av']-th_ln_noise[4]['Sb_th'])/th_ln_noise[4]['Sb_th']*100, ":", lw=2, color=colors[4], label=label[4] + " noise")
    l6, = ax2.plot(ln_noise[5]['T'], abs(ln_noise[5]['Sb_av']-th_ln_noise[5]['Sb_th'])/th_ln_noise[5]['Sb_th']*100, ":", lw=2, color=colors[5], label=label[5] + " noise")
    l7, = ax2.plot(ln_noise[6]['T'], abs(ln_noise[6]['Sb_av']-th_ln_noise[6]['Sb_th'])/th_ln_noise[6]['Sb_th']*100, ":", lw=2, color=colors[6], label=label[6] + " noise")
    l8, = ax2.plot(ln_noise[7]['T'], abs(ln_noise[7]['Sb_av']-th_ln_noise[7]['Sb_th'])/th_ln_noise[7]['Sb_th']*100, ":", lw=2, color=colors[7], label=label[7] + " noise")
    
    ax2.set_ylabel(r"$\dfrac{\langle S_b \rangle - \langle S_{b} \rangle ^{th}}{\langle S_{b} \rangle ^{th}}\quad$[%] ", fontsize=tick_fs)
    ax2.set_xlabel(r'$\mathcal{T}$ training patterns', fontsize=tick_fs)
    ax2.tick_params(labelsize=tick_fs)
    if(sat==False):
        ax2.set_ylim(0, 0.06)
    ax2.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax2.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    #ax2.legend(fontsize=legend_fs, framealpha=0.75, ncol=2)


    ax3.text(-0.1, 1.05, "C", weight="bold", fontsize=30, color='k', transform=ax3.transAxes)
    for i in range(len(ln_noise)):
        ax3.plot(ln_noise[i]['T'], th_ln_noise[i]['varSb_th'], "--", color=th_colors[i], label="Th - " + label[i] + " noise")
        ax3.plot(ln_noise[i]['T'], ln_noise[i]['varSb_av'], "^", markersize=7, color=colors[i], label="Sim - " + label[i] + " noise")
    ax3.set_ylabel(r"$\sigma^2_b$ $\quad [\mathrm{pA}^2 \times \mathrm{Hz}^2]$", fontsize=tick_fs)
    ax3.tick_params(labelsize=tick_fs)
    ax3.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax3.set_xticklabels([])


    for i in range(len(ln_noise)):
        ax4.plot(ln_noise[i]['T'], abs(ln_noise[i]['varSb_av']-th_ln_noise[i]['varSb_th'])/th_ln_noise[i]['varSb_th']*100, ":", lw=2, color=colors[i], label=label[i] + " noise")    
    ax4.set_ylabel(r"$\dfrac{\sigma^2_b - \sigma^2_{b^{th}}}{\sigma^2_{b^{th}}}\quad$[%]", fontsize=tick_fs)
    ax4.set_xlabel(r'$\mathcal{T}$ training patterns', fontsize=tick_fs)
    ax4.tick_params(labelsize=tick_fs)
    ax4.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax4.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])


    ax5.text(-0.1, 1.05, "B", weight="bold", fontsize=30, color='k', transform=ax5.transAxes)
    for i in range(len(ln_noise)):
        ax5.plot(ln_noise[i]['T'], th_ln_noise[i]['Sc_th'], "--", color=th_colors[i], label="Th - " + label[i] + " noise")
        ax5.plot(ln_noise[i]['T'], ln_noise[i]['Sc_av'], "^", markersize=7, color=colors[i], label="Sim - " + label[i] + " noise")
        
    ax5.set_ylabel(r"$\langle S_c \rangle$ [pA $\times$ Hz]", fontsize=tick_fs)
    ax5.tick_params(labelsize=tick_fs)
    #ax5.set_xscale('log')
    ax5.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax5.set_xticklabels([])

    for i in range(len(ln_noise)):
        ax6.plot(ln_noise[i]['T'], abs(ln_noise[i]['Sc_av']-th_ln_noise[i]['Sc_th'])/th_ln_noise[i]['Sc_th']*100, ":", lw=2, color=colors[i], label=label[i] + " noise")

    ax6.set_ylabel(r"$\dfrac{\langle S_{c} \rangle - \langle S_{c}\rangle ^{th}}{\langle S_{c}\rangle ^{th}}\quad$[%]", fontsize=tick_fs)
    ax6.set_xlabel(r'$\mathcal{T}$ training patterns', fontsize=tick_fs)
    ax6.tick_params(labelsize=tick_fs)
    ax6.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax6.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])


    for i in range(len(ln_noise)):
        ax7.plot(ln_noise[i]['T'], np.abs(th_ln_noise[i]['Sc_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']), "--", color=th_colors[i], label='_nolegend_')
        ax7.plot(ln_noise[i]['T'], np.abs(ln_noise[i]['Sc_av']-ln_noise[i]['Sb_av'])/np.sqrt(ln_noise[i]['varSb_av']), "^", markersize=7, color=colors[i], label='_nolegend_')
    

    ax7.plot(np.linspace(5000,100000,5),SDNR_thr*np.ones(5), linestyle='-',color='coral')
    ax7.text(80000, y=SDNR_thr+0.2, color='coral', fontsize=legend_fs+2, s=r'SDNR$_{thr}$')


    ax7.text(-0.1, 1.05, "D", weight="bold", fontsize=30, color='k', transform=ax7.transAxes)
    ax7.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax7.tick_params(labelsize=tick_fs)
    #ax7.set_xscale('log')
    ax7.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax7.set_xticklabels([])
    
    for i in range(len(ln_noise)):
        ax8.plot(ln_noise[i]['T'], abs(np.abs(ln_noise[i]['Sc_av']-ln_noise[i]['Sb_av'])/np.sqrt(ln_noise[i]['varSb_av'])-np.abs(th_ln_noise[i]['Sc_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']))/(np.abs(th_ln_noise[i]['Sc_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']))*100, ":", lw=2, color=colors[i], label=label[i] + " noise")
    
    ax8.set_ylabel(r"$\dfrac{SDNR - SDNR^{th}}{SDNR^{th}}\quad$[%] ", fontsize=tick_fs)
    ax8.set_xlabel(r'$\mathcal{T}$ training patterns', fontsize=tick_fs)
    ax8.tick_params(labelsize=tick_fs)
    ax8.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax8.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])

    fig1.savefig('lognormal' + saturation +'.png', dpi=300)


def plot_SDNR_sat_nsat(ln_noise, th_ln_noise, ln_noise_sat, th_lnoise_sat, label, prob_thr):

    def SDNR_thres(prob_thr):
        
        SDNR_thr=erfinv(2*prob_thr-1) *2*np.sqrt(2)
        return(SDNR_thr)
    
    SDNR_thr=SDNR_thres(prob_thr)
    print('Having P_C=' + str(prob_thr)+ ' the SDNR threshold is ' + str(SDNR_thr))

    """
    Plot data and saves the plot

    Parameters
    ----------
    ln, th_ln: pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and Sc for lognormal rate model
    ln_noise, th_ln_noise: list of pandas DataFrame
        DataFrame with simulation and theoretical values of Sb, varSb and Sc for lognormal rate model with noise

    """

    # fontsize params
    legend_fs =20
    tick_fs = 25

    #width and height ratios of subplots
    widths= [1, 1]
    heights= [3, 1]

    fig1, axs1 = plt.subplots(ncols=2, nrows=2, figsize = (20,18),constrained_layout=True,gridspec_kw={'height_ratios': heights})
  

    #subplots
    ax1 = axs1[0,0] # SDNR - non saturated
    ax2 = axs1[1,0] # residue - SDNR - non saturated
    ax3 = axs1[0,1] # SDNR - saturated
    ax4 = axs1[1,1] # residue - SDNR - saturated


    #parameters of grid
    y_begin=0.22
    x_begin=0.12
    height_res=0.1
    height_dat=0.22
    space_s=0.01
    space_l=0.065
    central_space=0.1
    h_space=0.5
    width_g=0.35


    ##########grid plot lognormal model##############

    
    ax1.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax2.set_position([x_begin, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])
    ax3.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l+height_res+space_s, width_g, height_dat])
    ax4.set_position([x_begin+width_g+central_space, y_begin+height_res+space_s+height_dat+space_l, width_g, height_res])


    ############################## PLOT LOGNORMAL MODEL #########################################################
    
    colors = ["magenta", "darkkhaki", "red", "blue", "black", "green"]
    th_colors = ["plum", "khaki", "orange", "cornflowerblue", "grey", "limegreen"]

    from matplotlib.legend_handler import HandlerTuple
        
    l1_th, = ax1.plot(ln_noise[0]['T'], np.abs(th_ln_noise[0]['Sc_th']-th_ln_noise[0]['Sb_th'])/np.sqrt(th_ln_noise[0]['varSb_th']), "--", color=th_colors[0], label='_nolegend_')
    l1, = ax1.plot(ln_noise[0]['T'], np.abs(ln_noise[0]['Sc_av']-ln_noise[0]['Sb_av'])/np.sqrt(ln_noise[0]['varSb_av']), "^", markersize=7, color=colors[0], label='_nolegend_')

    l2_th, = ax1.plot(ln_noise[1]['T'], np.abs(th_ln_noise[1]['Sc_th']-th_ln_noise[1]['Sb_th'])/np.sqrt(th_ln_noise[1]['varSb_th']), "--", color=th_colors[1], label='_nolegend_')
    l2, = ax1.plot(ln_noise[1]['T'], np.abs(ln_noise[1]['Sc_av']-ln_noise[1]['Sb_av'])/np.sqrt(ln_noise[1]['varSb_av']), "^", markersize=7, color=colors[1], label='_nolegend_')

    l3_th, = ax1.plot(ln_noise[2]['T'], np.abs(th_ln_noise[2]['Sc_th']-th_ln_noise[2]['Sb_th'])/np.sqrt(th_ln_noise[2]['varSb_th']), "--", color=th_colors[2], label='_nolegend_')
    l3, = ax1.plot(ln_noise[2]['T'], np.abs(ln_noise[2]['Sc_av']-ln_noise[2]['Sb_av'])/np.sqrt(ln_noise[2]['varSb_av']), "^", markersize=7, color=colors[2], label='_nolegend_')

    l4_th, = ax1.plot(ln_noise[3]['T'], np.abs(th_ln_noise[3]['Sc_th']-th_ln_noise[3]['Sb_th'])/np.sqrt(th_ln_noise[3]['varSb_th']), "--", color=th_colors[3], label='_nolegend_')
    l4, = ax1.plot(ln_noise[3]['T'], np.abs(ln_noise[3]['Sc_av']-ln_noise[3]['Sb_av'])/np.sqrt(ln_noise[3]['varSb_av']), "^", markersize=7, color=colors[3], label='_nolegend_')

    l5_th, = ax1.plot(ln_noise[4]['T'], np.abs(th_ln_noise[4]['Sc_th']-th_ln_noise[4]['Sb_th'])/np.sqrt(th_ln_noise[4]['varSb_th']), "--", color=th_colors[4], label='_nolegend_')
    l5, = ax1.plot(ln_noise[4]['T'], np.abs(ln_noise[4]['Sc_av']-ln_noise[4]['Sb_av'])/np.sqrt(ln_noise[4]['varSb_av']), "^", markersize=7, color=colors[4], label='_nolegend_')
    
    l6_th, = ax1.plot(ln_noise[5]['T'], np.abs(th_ln_noise[5]['Sc_th']-th_ln_noise[5]['Sb_th'])/np.sqrt(th_ln_noise[5]['varSb_th']), "--", color=th_colors[5], label='_nolegend_')
    l6, = ax1.plot(ln_noise[5]['T'], np.abs(ln_noise[5]['Sc_av']-ln_noise[5]['Sb_av'])/np.sqrt(ln_noise[5]['varSb_av']), "^", markersize=7, color=colors[5], label='_nolegend_')
    
    ax1.plot(np.linspace(5000,100000,5),SDNR_thr*np.ones(5), linestyle='-',color='coral')
    ax1.text(80000, y=SDNR_thr+0.2, color='coral', fontsize=legend_fs+2, s=r'SDNR$_{thr}$')
    
    ax1.text(-0.1, 1.05, "A", weight="bold", fontsize=30, color='k', transform=ax1.transAxes)
    ax1.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax1.set_xticklabels([])
    ax4.legend(handles =[(plt.scatter([], [], marker='^', s=100, color=l1.get_color()),plt.plot([], linestyle='--', lw=2, color=l1_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l1_th.get_color())[0]),
                (plt.scatter([], [], marker='^', s=100, color=l2.get_color()),plt.plot([], linestyle='--', lw=2, color=l2_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l2_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l3.get_color()),plt.plot([], linestyle='--', lw=2, color=l3_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l3_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l4.get_color()),plt.plot([], linestyle='--', lw=2, color=l4_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l4_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l5.get_color()),plt.plot([], linestyle='--', lw=2, color=l5_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l5_th.get_color())[0]), 
                (plt.scatter([], [], marker='^', s=100, color=l6.get_color()),plt.plot([], linestyle='--', lw=2, color=l6_th.get_color())[0],plt.plot([], linestyle=':', lw=3, color=l6_th.get_color())[0])], 
                
                labels = ['No noise', '1 Hz std noise', '2 Hz std noise', '3 Hz std noise', '4 Hz std noise', '5 Hz std noise'], numpoints=6, ncol=2,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_fs+2, framealpha=1.0, handlelength=5, title=r"Simulation ($\blacktriangle$), theoretical estimation (--) and comparison ($\cdots$)", title_fontsize=legend_fs+5,
              bbox_to_anchor=(0.63, -1.5),loc='center right')

    
    for i in range(len(ln_noise)):
        ax2.plot(ln_noise[i]['T'], abs(np.abs(ln_noise[i]['Sc_av']-ln_noise[i]['Sb_av'])/np.sqrt(ln_noise[i]['varSb_av'])-np.abs(th_ln_noise[i]['Sc_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']))/(np.abs(th_ln_noise[i]['Sc_th']-th_ln_noise[i]['Sb_th'])/np.sqrt(th_ln_noise[i]['varSb_th']))*100, ":", lw=2, color=colors[i], label='_nolegend_')
    
    ax2.set_ylabel(r"$\dfrac{SDNR - SDNR^{th}}{SDNR^{th}}\quad$[%] ", fontsize=tick_fs)
    ax2.set_xlabel(r'$\mathcal{T}$ training patterns', fontsize=tick_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax2.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])
    #ax2.legend(fontsize=legend_fs, framealpha=0.75, ncol=2)


    for i in range(len(ln_noise)):
        ax3.plot(ln_noise_sat[i]['T'], np.abs(th_lnoise_sat[i]['Sc_th']-th_lnoise_sat[i]['Sb_th'])/np.sqrt(th_lnoise_sat[i]['varSb_th']), "--", color=th_colors[i], label='_nolegend_')
        ax3.plot(ln_noise_sat[i]['T'], np.abs(ln_noise_sat[i]['Sc_av']-ln_noise_sat[i]['Sb_av'])/np.sqrt(ln_noise_sat[i]['varSb_av']), "^", markersize=7, color=colors[i], label='_nolegend_')
    

    ax3.plot(np.linspace(5000,100000,5),SDNR_thr*np.ones(5), linestyle='-',color='coral')
    ax3.text(80000, y=SDNR_thr+0.2, color='coral', fontsize=legend_fs+2, s=r'SDNR$_{thr}$')


    ax3.text(-0.1, 1.05, "B", weight="bold", fontsize=30, color='k', transform=ax3.transAxes)
    ax3.set_ylabel(r"SDNR", fontsize=tick_fs)
    ax3.tick_params(labelsize=tick_fs)
    #ax7.set_xscale('log')
    ax3.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax3.set_xticklabels([])
    
    for i in range(len(ln_noise)):
        ax4.plot(ln_noise_sat[i]['T'], abs(np.abs(ln_noise_sat[i]['Sc_av']-ln_noise_sat[i]['Sb_av'])/np.sqrt(ln_noise_sat[i]['varSb_av'])-np.abs(th_lnoise_sat[i]['Sc_th']-th_lnoise_sat[i]['Sb_th'])/np.sqrt(th_lnoise_sat[i]['varSb_th']))/(np.abs(th_lnoise_sat[i]['Sc_th']-th_lnoise_sat[i]['Sb_th'])/np.sqrt(th_lnoise_sat[i]['varSb_th']))*100, ":", lw=2, color=colors[i], label=label[i] + " noise")
    
    ax4.set_ylabel(r"$\dfrac{SDNR - SDNR^{th}}{SDNR^{th}}\quad$[%] ", fontsize=tick_fs)
    ax4.set_xlabel(r'$\mathcal{T}$ training patterns', fontsize=tick_fs)
    ax4.tick_params(labelsize=tick_fs)
    ax4.set_xticks([5000, 25000, 50000, 75000, 100000])
    ax4.set_xticklabels(["5K", "25K", "50K", "75K", "100K"])


    ax1.set_title("No rate correction", fontsize=tick_fs)
    ax3.set_title("Saturation of firing rates", fontsize=tick_fs)

    ax1.set_ylim(1, 7)
    ax3.set_ylim(1, 7)




    fig1.savefig('SDNR_comparison' +'.png', dpi=300)

    plt.show()







    
saturation = True
if saturation==True:
    sat = "_saturation"
else:
    sat = ""

# values extracted from simulations
ln_rate = get_data("../simulations/no_noise_simulations", 10)
ln_rate_noise1 = get_data("../simulations/noise_0.2Hz_simulations"+sat, 10)
ln_rate_noise2 = get_data("../simulations/noise_0.4Hz_simulations"+sat, 10)
ln_rate_noise3 = get_data("../simulations/noise_0.6Hz_simulations"+sat, 10)
ln_rate_noise4 = get_data("../simulations/noise_0.8Hz_simulations"+sat, 10)
ln_rate_noise5 = get_data("../simulations/noise_1Hz_simulations"+sat, 10)
ln_rate_noise6 = get_data("../simulations/noise_1.5Hz_simulations"+sat, 10)
ln_rate_noise7 = get_data("../simulations/noise_2Hz_simulations"+sat, 10)
ln_rate_noise8 = get_data("../simulations/noise_3Hz_simulations"+sat, 10)
ln_rate_noise9 = get_data("../simulations/noise_4Hz_simulations"+sat, 10)
ln_rate_noise10 = get_data("../simulations/noise_5Hz_simulations"+sat, 10)


# theoretical values
th_ln = get_th_data("../simulations/no_noise_simulations")
th_ln_noise1 = get_th_data("../simulations/noise_0.2Hz_simulations"+sat)
th_ln_noise2 = get_th_data("../simulations/noise_0.4Hz_simulations"+sat)
th_ln_noise3 = get_th_data("../simulations/noise_0.6Hz_simulations"+sat)
th_ln_noise4 = get_th_data("../simulations/noise_0.8Hz_simulations"+sat)
th_ln_noise5 = get_th_data("../simulations/noise_1Hz_simulations"+sat)
th_ln_noise6 = get_th_data("../simulations/noise_1.5Hz_simulations"+sat)
th_ln_noise7 = get_th_data("../simulations/noise_2Hz_simulations"+sat)
th_ln_noise8 = get_th_data("../simulations/noise_3Hz_simulations"+sat)
th_ln_noise9 = get_th_data("../simulations/noise_4Hz_simulations"+sat)
th_ln_noise10 = get_th_data("../simulations/noise_5Hz_simulations"+sat)

prob_thr=0.95
ln_rate_noise = [ln_rate, ln_rate_noise1, ln_rate_noise2, ln_rate_noise3, ln_rate_noise4, ln_rate_noise5, ln_rate_noise6, ln_rate_noise7]
th_ln_noise = [th_ln, th_ln_noise1, th_ln_noise2, th_ln_noise3, th_ln_noise4, th_ln_noise5, th_ln_noise6, th_ln_noise7]
label = ["No", "0.2 Hz", "0.4 Hz", "0.6 Hz", "0.8 Hz", "1 Hz", "1.5 Hz", "2 Hz"]


## appendix plot ##

if saturation:
    app_plot_ln_sat = [ln_rate, ln_rate_noise5, ln_rate_noise7, ln_rate_noise8, ln_rate_noise9, ln_rate_noise10]
    th_app_plot_ln_sat = [th_ln, th_ln_noise5, th_ln_noise7, th_ln_noise8, th_ln_noise9, th_ln_noise10]

    # load the data without saturation
    new_ln_rate_noise5 = get_data("../simulations/noise_1Hz_simulations", 10)
    new_ln_rate_noise7 = get_data("../simulations/noise_2Hz_simulations", 10)
    new_ln_rate_noise8 = get_data("../simulations/noise_3Hz_simulations", 10)
    new_ln_rate_noise9 = get_data("../simulations/noise_4Hz_simulations", 10)
    new_ln_rate_noise10 = get_data("../simulations/noise_5Hz_simulations", 10)

    new_th_ln_noise5 = get_th_data("../simulations/noise_1Hz_simulations")
    new_th_ln_noise7 = get_th_data("../simulations/noise_2Hz_simulations")
    new_th_ln_noise8 = get_th_data("../simulations/noise_3Hz_simulations")
    new_th_ln_noise9 = get_th_data("../simulations/noise_4Hz_simulations")
    new_th_ln_noise10 = get_th_data("../simulations/noise_5Hz_simulations")

    app_plot_ln = [ln_rate, new_ln_rate_noise5, new_ln_rate_noise7, new_ln_rate_noise8, new_ln_rate_noise9, new_ln_rate_noise10]
    th_app_plot_ln = [th_ln, new_th_ln_noise5, new_th_ln_noise7, new_th_ln_noise8, new_th_ln_noise9, new_th_ln_noise10]

else:
    app_plot_ln = [ln_rate, ln_rate_noise5, ln_rate_noise7, ln_rate_noise8, ln_rate_noise9, ln_rate_noise10]
    th_app_plot_ln = [th_ln, th_ln_noise5, th_ln_noise7, th_ln_noise8, th_ln_noise9, th_ln_noise10]

    # load the data without saturation
    new_ln_rate_noise5 = get_data("../simulations/noise_1Hz_simulations_saturation", 10)
    new_ln_rate_noise7 = get_data("../simulations/noise_2Hz_simulations_saturation", 10)
    new_ln_rate_noise8 = get_data("../simulations/noise_3Hz_simulations_saturation", 10)
    new_ln_rate_noise9 = get_data("../simulations/noise_4Hz_simulations_saturation", 10)
    new_ln_rate_noise10 = get_data("../simulations/noise_5Hz_simulations_saturation", 10)

    new_th_ln_noise5 = get_th_data("../simulations/noise_1Hz_simulations_saturation")
    new_th_ln_noise7 = get_th_data("../simulations/noise_2Hz_simulations_saturation")
    new_th_ln_noise8 = get_th_data("../simulations/noise_3Hz_simulations_saturation")
    new_th_ln_noise9 = get_th_data("../simulations/noise_4Hz_simulations_saturation")
    new_th_ln_noise10 = get_th_data("../simulations/noise_5Hz_simulations_saturation")

    app_plot_ln_sat = [ln_rate, new_ln_rate_noise5, new_ln_rate_noise7, new_ln_rate_noise8, new_ln_rate_noise9, new_ln_rate_noise10]
    th_app_plot_ln_sat = [th_ln, new_th_ln_noise5, new_th_ln_noise7, new_th_ln_noise8, new_th_ln_noise9, new_th_ln_noise10]


app_label = ["No", "1 Hz", "2 Hz", "3 Hz", "4 Hz", "5 Hz"]


#plot_data(ln_rate_noise, th_ln_noise, label, sat,prob_thr)
plot_SDNR_sat_nsat(app_plot_ln, th_app_plot_ln, app_plot_ln_sat, th_app_plot_ln_sat, app_label, prob_thr)





"""
def sqrt_function(X, a):
    return a / np.sqrt(X) 
sdnr_thr = erfinv(2*prob_thr-1)*np.sqrt(8)
sdnr_noise0 = np.asarray([(ln_rate['Sc_av'][i]-ln_rate['Sb_av'][i])/np.sqrt(ln_rate['varSb_av'][i]) for i in range(len(ln_rate['T']))])
sdnr_noise1 = np.asarray([(ln_rate_noise1['Sc_av'][i]-ln_rate_noise1['Sb_av'][i])/np.sqrt(ln_rate_noise1['varSb_av'][i]) for i in range(len(ln_rate_noise1['T']))])
#sdnr_noise2 = np.asarray([(ln_rate_noise2['Sc_av'][i]-ln_rate_noise2['Sb_av'][i])/np.sqrt(ln_rate_noise2['varSb_av'][i]) for i in range(len(ln_rate_noise2['T']))])
#sdnr_noise3 = np.asarray([(ln_rate_noise3['Sc_av'][i]-ln_rate_noise3['Sb_av'][i])/np.sqrt(ln_rate_noise3['varSb_av'][i]) for i in range(len(ln_rate_noise3['T']))])
#sdnr_noise4 = np.asarray([(ln_rate_noise4['Sc_av'][i]-ln_rate_noise4['Sb_av'][i])/np.sqrt(ln_rate_noise4['varSb_av'][i]) for i in range(len(ln_rate_noise4['T']))])
sdnr_noise5 = np.asarray([(ln_rate_noise5['Sc_av'][i]-ln_rate_noise5['Sb_av'][i])/np.sqrt(ln_rate_noise5['varSb_av'][i]) for i in range(len(ln_rate_noise5['T']))])

T = ln_rate_noise1['T']

# Generate sample data
# Fit the data to the sqrt_function
params0, covariance0 = curve_fit(sqrt_function, T, sdnr_noise0)
params1, covariance1 = curve_fit(sqrt_function, T, sdnr_noise1)
#params2, covariance2 = curve_fit(sqrt_function, T, sdnr_noise2)
#params3, covariance3 = curve_fit(sqrt_function, T, sdnr_noise3)
#params4, covariance4 = curve_fit(sqrt_function, T, sdnr_noise4)
params5, covariance5 = curve_fit(sqrt_function, T, sdnr_noise5)

# Extract the fitted parameter
a_fit_noise0 = params0[0]
a_fit_noise1 = params1[0]
#print(a_fit_noise1)
#b_fit_noise1 = params1[1]
#print(b_fit_noise1)
#a_fit_noise2 = params2[0]
#b_fit_noise2 = params2[1]
#a_fit_noise3 = params3[0]
#b_fit_noise3 = params3[1]
#a_fit_noise4 = params4[0]
#b_fit_noise4 = params4[1]
a_fit_noise5 = params5[0]
T_max_noise0=a_fit_noise0**2/(sdnr_thr)**2
T_max_noise1=a_fit_noise1**2/(sdnr_thr)**2
T_max_noise5=a_fit_noise5**2/(sdnr_thr)**2


print("Massimo numero di pattern con 0 Hz noise: " + str(T_max_noise0)+ '\n')
print("Massimo numero di pattern con 1 Hz noise: " + str(T_max_noise1)+ '\n')
print("Massimo numero di pattern con 5 Hz noise: " + str(T_max_noise5)+ '\n')
"""



plt.show()
