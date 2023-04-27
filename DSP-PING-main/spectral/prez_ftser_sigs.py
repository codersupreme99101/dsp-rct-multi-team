import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

pathway= "outputs_ftdomain/"

#start

f_s=10**6
t_max=4
t=np.arange(int(t_max*f_s)) #s 
s=np.zeros(len(t))

fac=2*1.414
f_arr=np.array([2500,5000,7500, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000]) #n val. 
a_arr=fac*np.random.rand(len(f_arr))

burst_period=20*10**3
calm_period=1*10**6

spacer_arr=np.zeros((len(t),))

for i in range(len(f_arr)):
    
    num_poss=int(np.ceil(len(t)/(burst_period+calm_period)))
    spacer=np.zeros((len(t),))
    spacer_l=spacer
    spacer_u=spacer

    
    for k in range(num_poss):
        
        spacer[(k*calm_period+k*burst_period):((k)*(calm_period)+(k+1)*(burst_period))]=1
                                
    spacer=np.roll(spacer, i*burst_period) 
    
    spacer[:i*burst_period]=0
    
    spacer_arr+=(f_arr[i]*spacer)
        
    noise_a=np.random.normal(0,1, size=(len(s), ))
    
    s+=(a_arr[i]*np.cos(f_arr[i]*spacer)+a_arr[i]*noise_a)
    
fx, tx, sx=sig.spectrogram(s, f_s)


fig = plt.figure(figsize = (18,14))

plt.plot(t,spacer_arr)
plt.ylabel("Freq. Hz.")
plt.xlabel("timesteps (s) @{} samples".format(f_s))
plt.savefig(pathway+"spectro_ftsplit_v1.png")

#end
