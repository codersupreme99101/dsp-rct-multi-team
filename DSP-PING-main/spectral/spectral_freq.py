import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

pathway="outputs_freqdomain/"

#start

f_s=100000
t_max=4
t=np.arange(int(t_max*f_s)) #s 
s=np.zeros(len(t))

fac=2*1.414
f_arr=[300, 50, 1000, 5000] #n val. 
a_arr=fac*np.random.rand(len(f_arr))

for i in range(len(f_arr)):
    noise_a=np.random.normal(0,1, size=(len(s), ))
    s+=(a_arr[i]*np.cos(f_arr[i]*t)+a_arr[i]*noise_a)
    
#plt.plot(t,s)

fx, tx, sx=sig.spectrogram(s, f_s)

plt.pcolormesh(tx,fx/4,sx, shading="gouraud")
plt.ylabel("Freq. Hz.")
plt.xlabel("timesteps (s) @{} samples".format(f_s))
plt.savefig(pathway+"spectro_v1.png")

#end
