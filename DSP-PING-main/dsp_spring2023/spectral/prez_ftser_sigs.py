import numpy as np
import matplotlib.pyplot as plt

pathway= "outputs_ftdomain/"

#start

f_s=10**6
t_max=4
t=np.arange(int(t_max*f_s)) #s 

fac=2*1.414
f_arr=np.array([12500,10000,25000,7500,50000])#np.array([12500,1000,25000,7500,50000])#np.array([2500,5000,7500, 10000])#, 12500, 15000, 20000, 30000, 50000, 75000, 100000]) #n val. 
a_arr=fac*np.random.rand(len(f_arr))

burst_period=20*10**3
calm_period=1*10**6

factor_calm=2
burst_period_arr=np.random.randint(1,3*t_max,len(f_arr))
calm_period_arr=np.random.randint(1,int(t_max/factor_calm)+1, len(f_arr)) #multiplied, integer due to each number being the 1us sample 

calm_period_j=(calm_period*calm_period_arr)
burst_period_j=(burst_period*burst_period_arr) #int for random value to integer

spacer_arr=[]

for i in range(len(f_arr)):
    
    num_poss=int(np.ceil(len(t)/(burst_period_j[i]+calm_period_j[i])))
    spacer=np.zeros((len(t),))
    spacer_l=spacer
    spacer_u=spacer

    
    for k in range(num_poss):
        
        spacer[(k*calm_period_j[i]+k*burst_period_j[i]):((k)*(calm_period_j[i])+(k+1)*(burst_period_j[i]))]=1
                                
    spacer=np.roll(spacer, i*burst_period_j[i]) 
    
    spacer[:i*burst_period_arr[i]*burst_period]=0
    
    spacer_arr.append(f_arr[i]*spacer)
                

#plot

color_plot=["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "cyan", "black", "deeppink", "blueviolet", "lime", "olive", "saddlebrown"]

fig = plt.figure(figsize = (18,14))

for j in range(len(spacer_arr)):
    plt.plot(t,spacer_arr[j], color=color_plot[j])
    
plt.title("Timesteps (1 per us) to Frequency (Hz) Analysis for {} transmitters".format(len(f_arr)))
plt.ylabel("Frequency (Hz.)")
plt.xlabel("Timesteps (1 per us) @{} samples".format(f_s))
plt.savefig(pathway+"spectro_ftsplit_v{}.png".format(np.random.random()))

#end
