import numpy as np
import scipy.signal as sig
from scipy import fftpack
import matplotlib.pyplot as plt

pathway="outputs_freqdomain/"

#start

f_s=10**6
t_max=4
t=np.arange(int(t_max*f_s)) #s 
s=np.zeros(len(t))

f_arr=[2500, 5000, 7500, 10000] #n val. 

for i in range(len(f_arr)):
    s+=(np.cos(2*np.pi*f_arr[i]*t))
    
#plt.plot(t,s)

'''
fx, tx, sx=sig.spectrogram(s, f_s)

plt.pcolormesh(tx,fx,sx, shading="gouraud")
plt.ylabel("Freq. Hz.")
plt.xlabel("timesteps (s) @{} samples".format(f_s))
plt.savefig(pathway+"spectro_v1.png")
'''
T=t
y=s
N=f_s

yf = fftpack.fft(y)


print(yf)

#end
