import numpy as np
data = np.loadtxt('all.dat')
var=data[:,3]
print(np.mean(var))
S2=data[:,2]
print(np.mean(S2))
Sb=data[:,1]
print(np.mean(Sb))
