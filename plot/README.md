# Plot reproduction

In order to reproduce the plots of the manuscript you can run the following Python scripts:

- [plot_cn.py](plot_cn.py): returns cn_plot.png, i.e., Figure 11
- [plot_rewiring_scaling.py](plot_rewiring_scaling.py): returns rewiring_scaling.png, i.e., Figure 6
- [plot_sim_th_comparison.py](plot_sim_th_comparison.py): returns lognormal.png, lognormal_saturation.png and SDNR_comparison.png, i.e., Figures 3, 4 and 10. Function ```plot_data``` returns Figures 3 or 4 depending on the value of the boolean ```saturation```, whereas function ```plot_SDNR_sat_nsat``` returns Figure 10
- [plot_variable_rewiring.py](plot_variable_rewiring.py): returns variable_rewiring.png, i.e., Figure 5
- [plot.py](plot.py): returns Figure 12


The plots use the data stored inside the [simulations](../simulations/) directory.


