import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

pathway= "outputs_ftdomain/"


'''
period, period variability.
'''

#start

f_s=10**6 #hyperparams
t_max=4 #constants universally called 
f_c=172*10**6
t=np.arange(int(t_max*f_s)) #s 
mcs=2500 #min chan sep in Hz
burst_period=20*10**3
calm_period=1*10**6#40*10**4 #2x burst for demo purposes #1*10**6 #real value
mean_amp=np.sqrt(2) #units 
mean_val_noise=np.sqrt(2*np.pi)

f_arr=np.array([2500,5000,7500, 10000])#np.array([2500,6500,9000, 13000])#np.array([12500,1000,25000,7500,45000])#np.array([2500,5000,7500, 10000])#np.array([12500,1000,25000,7500,50000])#np.array([2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500])#np.array([2500,5000,7500, 10000])#np.array([12500,1000,25000,7500,50000])#np.array([2500,5000,7500, 10000])#, 12500, 15000, 20000, 30000, 50000, 75000, 100000]) #n val. 

def check_minchansep(f_arr): #check min chan sep between all
    
    for i in range(len(f_arr)):
        for j in range(len(f_arr)):
            if i!=j:
                if abs(f_arr[i]-f_arr[j])<mcs:
                    return False
    
    return True

def plot_ft_dom(t, spacer_arr): #make plot 
    
    color_plot=["blue", "aqua", "green", "red", "purple", "brown", "pink", "tomato", "cyan", "black", "deeppink", "blueviolet", "lime", "olive", "saddlebrown"]

    fig = plt.figure(figsize = (18,14))
    plt.rcParams['axes.facecolor'] = 'whitesmoke' #to make the colors pop out

    for j in range(len(spacer_arr)):
        plt.plot(t,spacer_arr[j], color=color_plot[j], label="transmitter {}".format(j))
        
    #plt.grid(True) #grid is messy
    plt.legend(loc="best")
    plt.title("Timesteps (1 per us) to Frequency (Hz) Analysis for {} transmitters".format(len(f_arr)))
    plt.ylabel("Frequency (Hz.)")
    plt.xlabel("Timesteps (1 per us) @{} samples".format(f_s))
    plt.savefig(pathway+"spectro_ftsplit_v{}.png".format(np.random.random()))
    
    pass

def make_signal(t, spacer_arr_temp): #make signal
        
    signal=np.zeros(spacer_arr_temp[0].shape, dtype=np.complex128)
    
    amplitude_arr=np.random.uniform(0.9*mean_amp, 1.1*mean_amp, len(spacer_arr_temp))
    noise_i=np.random.uniform(-1*mean_val_noise, 1*mean_val_noise, spacer_arr_temp.shape)
        
    for i in range(len(spacer_arr_temp)):
        spacer_i=spacer_arr_temp[i]
        spacer_i[spacer_i==np.nan]=0
        signal+=((amplitude_arr[i]*(np.cos(2*np.pi*t*f_c)+1j*np.sin(spacer_i)))+noise_i[i]) #check if real or complex, current data suggest complex

    return signal
                
def make_spacer_ft(t, f_arr, burst_period_j, calm_period_j, random_offset_period_j): #spacer array for f domain and t domain holder in binary amp . o ff
    
    spacer_arr=[]
    spacer_arr_temp=[]

    for i in range(len(f_arr)):
        
        num_poss=int(np.ceil(len(t)/(burst_period_j[i]+calm_period_j[i])))
        spacer=np.zeros((len(t),))

        for k in range(num_poss):
            
            spacer[int(k*calm_period_j[i]+k*burst_period_j[i]):int((k)*(calm_period_j[i])+(k+1)*(burst_period_j[i]))]=1
                                    
        spacer=np.roll(spacer, int(random_offset_period_j[i])) 
        
        spacer[:int(random_offset_period_j[i])]=0 #offset, so that the transmitter may have delay from its own unique pattern 
        
        spacer_ij=f_arr[i]*spacer
        spacer_arr_temp.append(spacer_ij) #for sig make 
        spacer_ij[spacer_ij==0]=np.nan #no dc offset plot, as there is none
        spacer_arr.append(spacer_ij) #add
    
    spacer_arr=np.array(spacer_arr)   
    
    return spacer_arr    
                
if check_minchansep(f_arr): #true condition
    
    print("Valid sequence")

    burst_period_arr=np.random.uniform(0.9, 1.1,len(f_arr)) #+/- 10% of the given mean data at norm. 1
    calm_period_arr=np.random.uniform(0.9, 1.1, len(f_arr)) #multiplied, integer due to each number being the 1us sample 

    calm_period_j=np.round(calm_period*calm_period_arr) #In sample space, at least 1 in f_s 
    burst_period_j=np.round(burst_period*burst_period_arr) #int for random value to integer

    random_offset_period_arr=np.random.uniform(0,1.1, len(f_arr)) #non to 110% usual
    random_offset_period_j=np.round(random_offset_period_arr*burst_period) #some multiple of the burst period, for some simulated realistic values 

    #calls
    spacer_arr=make_spacer_ft(t, f_arr, burst_period_j, calm_period_j, random_offset_period_j)
    signal_final=make_signal(t, spacer_arr)        

    plot_ft_dom(t, spacer_arr)
    #use signal_final here for analysis later 

else:
    
    print("Invalid frequency set.")

#end
