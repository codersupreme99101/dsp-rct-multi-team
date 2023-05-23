import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import datetime
from matplotlib import cm
#imports 

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

idx_ch=0 #index value for f_s_a

f_super_arr=np.array([[12500,10000,25000,7500,30000], 
            [2500,5000,7500, 10000],
            [2500,6500,9000, 13000], 
            [12500,1000,25000,7500,45000], 
            [2500,5000,7500, 10000], 
            [12500,1000,25000,7500,50000], 
            [12500, 15000, 20000, 30000, 50000, 75000, 100000], 
            [12500,1000,25000,7500,50000], 
            [2500,5000,7500, 1000], 
            [2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500]])

f_arr=f_super_arr[idx_ch]

def check_minchansep(f_arr): #check min chan sep between all
    
    for i in range(len(f_arr)):
        for j in range(len(f_arr)):
            if i!=j:
                if abs(f_arr[i]-f_arr[j])<mcs:
                    return False
    
    return True

def make_signal(t, spacer_arr_temp): #make signal
        
    signal=np.zeros(spacer_arr_temp[0].shape, dtype=np.complex128)
    noise_2d=[]
    x_2d=[]
    amplitude_arr=np.random.uniform(0.9*mean_amp, 1.1*mean_amp, len(spacer_arr_temp))
    noise_i=np.random.uniform(-1*mean_val_noise, 1*mean_val_noise, spacer_arr_temp.shape)

        
    for i in range(len(spacer_arr_temp)):
        spacer_i=spacer_arr_temp[i]
        spacer_i[spacer_i==np.nan]=0
        signal+=((amplitude_arr[i]*(np.cos(2*np.pi*t*f_c)+1j*np.sin(spacer_i)))+noise_i[i]) #check if real or complex, current data suggest complex

        noise_2d.append(noise_i)
        x_2d.append(amplitude_arr[i]*spacer_i)

    return signal, np.array(noise_2d),np.array(x_2d)
                
def make_spacer_ft(t, f_arr, burst_period_j, calm_period_j, random_offset_period_j): #spacer array for f domain and t domain holder in binary amp . o ff
    
    spacer_arr=[]
    spacer_arr_temp=[]
    phi_2d=[]

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

        phi_2d.append((2*np.pi*f_arr[i]*2*np.pi*i/len(f_arr))+f_arr[0])
    
    spacer_arr=np.array(spacer_arr)   
    
    return spacer_arr,np.array(phi_2d)

def mle_multi_cfo(signal, sphi_k, t_error_tdx, x_b): #mle method

    y=signal.T
    cfo_arr=[]
    p=[]

    for k in range(len(sphi_k)):
        phi_k=np.diag(np.exp(1j*sphi_k[k]))
        a_k=np.zeros((len(signal), t_error_tdx))

        for i in range(t_error_tdx):
            x_bi=np.roll(x_b, i)
            x_bj=x_b[len(x_b)-i: ]
            x_bi[:i]=x_bj
            a_k.T[i]=x_bi
        p.append(np.matmul(phi_k, a_k))
    
    p=np.array(p)
    solution=np.matmul(np.matmul(np.matmul(y.H, np.matmul(p, np.linalg.inv(np.matmul(p.H, p)))), p.H), y)
    cfo_arr=np.argmax(solution,axis=1)

    return cfo_arr

def rmce(signal, sphi_k, t_error_tdx, x_b): #rmce method for mcfo

    y=signal.T
    cfo_arr=[]
    p=[]

    for k in range(len(sphi_k)):
        phi_k=np.diag(np.exp(1j*sphi_k[k]))
        a_k=np.zeros((len(signal), t_error_tdx))

        for i in range(t_error_tdx):
            x_bi=np.roll(x_b, i)
            x_bj=x_b[len(x_b)-i: ]
            x_bi[:i]=x_bj
            a_k.T[i]=x_bi
        p.append(np.matmul(phi_k, a_k))
    
    p=np.array(p)
    solution=np.matmul(np.matmul(y.H, np.matmul(p, p.H)), y)
    cfo_arr=np.argmax(solution/len(signal),axis=1)

    return cfo_arr
               
def sf_park_cfo(signal, offset, x_b): #sf park method 

    cfo_arr=[]

    for s in range(len(x_b)):

        conv_sig=0
        r_s=0
        p_s=0

        for i in range(offset/2):

            p_s+=(signal(int((s-i)%len(signal)))*signal(int((s+i)%len(signal))))
            r_s+=np.abs(signal(int((s+i)%len(signal))))**2

        for i in range(offset):

            m_s=np.abs(p_s**2)/r_s**2
            s_bar=np.argmax(m_s)-len(signal)/2
            a=int((s_bar-i)%len(signal))
            a_dash=int((s_bar+len(signal)-i)%len(signal))
            sig=signal[a]
            sig_star=np.conjugate(signal[a_dash])
            conv_sig+=(sig*sig_star)

        cfo_arr.append(np.angle(conv_sig))

    return np.array(cfo_arr)

def mimo_ofdm_cfo(signal, offset, m_value, x_b, sphi_k): #mimo ofdm method

    t_l=np.zeros((len(x_b), ))
    for t in range(offset):
        bpq=np.zeros((len(x_b), ))
        for p in range(len(x_b)):
            for k in range(len(x_b)):
                m_n_t=0
                for n in range(len(m_value)):
                    r_q=np.mean(signal(n)-np.conjugate(signal((n-t)%len(signal))))
                    m_n_t+=(r_q*np.exp(-t*(1j*2*np.pi*k*n/m_value)))
            bpq+=(x_b[p]*np.exp(1j*2*np.pi*sphi_k[p])-m_n_t)
        t_l+=(bpq**2)  

    return np.argmin(t_l, axis=1)

if check_minchansep(f_arr): #true condition
    
    print("Valid sequence. Proceeded.")

    burst_period_arr=np.random.uniform(0.9, 1.1,len(f_arr)) #+/- 10% of the given mean data at norm. 1
    calm_period_arr=np.random.uniform(0.9, 1.1, len(f_arr)) #multiplied, integer due to each number being the 1us sample 

    calm_period_j=np.round(calm_period*calm_period_arr) #In sample space, at least 1 in f_s 
    burst_period_j=np.round(burst_period*burst_period_arr) #int for random value to integer

    random_offset_period_arr=np.random.uniform(0,1.1, len(f_arr)) #non to 110% usual
    random_offset_period_j=np.round(random_offset_period_arr*burst_period) #some multiple of the burst period, for some simulated realistic values 

    #calls
    spacer_arr, p2d=make_spacer_ft(t, f_arr, burst_period_j, calm_period_j, random_offset_period_j)
    signal_final, noise_2d, x2d =make_signal(t, spacer_arr)        
    #use signal_final here for analysis later 

    m_tdx=20
    m_m_v=20 #hyperp
    m_offs=20

    abs_error_c1=[]
    abs_error_c2=[]
    abs_error_c3=[]

    mse_error_c1=[]
    mse_error_c2=[]
    mse_error_c3=[] #all arrs 

    mpe_error_c1=[]
    mpe_error_c2=[]
    mpe_error_c3=[]

    full_abs_error_c4=[]
    full_mse_error_c4=[]
    full_mpe_error_c4=[]

    for tdx in range(m_tdx): #for cfo methods 1 and 2 

        cfo1=mle_multi_cfo(signal_final, p2d, tdx, x2d)
        cfo2=rmce(signal_final, p2d, tdx, x2d)

        abs_error_c1.append(np.mean(np.abs(f_arr-cfo1)))
        mse_error_c1.append(np.mean((f_arr-cfo1)**2))
        mpe_error_c1.append(np.mean(np.abs(f_arr-cfo1)/f_arr))

        abs_error_c2.append(np.mean(np.abs(f_arr-cfo2)))
        mse_error_c2.append(np.mean((f_arr-cfo2)**2))
        mpe_error_c2.append(np.mean(np.abs(f_arr-cfo2)/f_arr))

    for m_v in range(m_m_v): #for cfo method 4

        mpe_error_c4=[]
        mse_error_c4=[]
        abs_error_c4=[]

        for offset in range(m_offs):

            cfo4=mimo_ofdm_cfo(signal_final, offset, m_v, x2d, p2d)

            abs_error_c4.append(np.mean(np.abs(f_arr-cfo4)))
            mse_error_c4.append((f_arr-cfo4)**2)
            mpe_error_c4.append(np.mean(np.abs(f_arr-cfo4)/f_arr))

        full_abs_error_c4.append(abs_error_c4)
        full_mse_error_c4.append(mse_error_c4)
        full_mpe_error_c4.append(mpe_error_c4)

    for offset in range(m_offs): #for cfo method 3

        cfo3=sf_park_cfo(signal_final, offset, x2d)

        abs_error_c3.append(np.mean(np.abs(f_arr-cfo3)))
        mse_error_c3.append((f_arr-cfo3)**2)
        mpe_error_c3.append(np.mean(np.abs(f_arr-cfo3)/f_arr))

    abs_error_c1=np.array(abs_error_c1)
    abs_error_c2=np.array(abs_error_c2)
    abs_error_c3=np.array(abs_error_c3)

    mse_error_c1=np.array(mse_error_c1)
    mse_error_c2=np.array(mse_error_c2)
    mse_error_c3=np.array(mse_error_c3) #all arrs 

    mpe_error_c1=np.array(mpe_error_c1)
    mpe_error_c2=np.array(mpe_error_c2)
    mpe_error_c3=np.array(mpe_error_c3)

    full_abs_error_c4=np.array(full_abs_error_c4)
    full_mse_error_c4=np.array(full_mse_error_c4)
    full_mpe_error_c4=np.array(full_mpe_error_c4)

    #plot and map:

    m_tdx_arr=np.arange(m_tdx)
    m_m_v_arr=np.arange(m_m_v) #for x axes and y axes, etc. 
    offs_arr=np.arange(m_offs)

    oo2d, mm2d =np.meshgrid(m_tdx_arr, offs_arr) #2d mesh 

    #plotting code 

    plt.figure(figsize=(12,10))
    plt.plot(m_tdx_arr, abs_error_c1, "ABS ERR MLE")
    plt.plot(m_tdx_arr, abs_error_c2, "ABS ERR RMCE")
    plt.title("Timing error (units) vs Absolute Error (CFO EST) (units)")
    plt.xlabel("Timing error (units)")
    plt.ylabel("Absolute Error (CFO EST) (units)")
    plt.savefig("outputs_ftdomain/abs_error_c12_{}.png".format(datetime.datetime.now()))

    plt.figure(figsize=(12,10))
    plt.plot(offs_arr, abs_error_c3, "ABS ERR SF PARK")
    plt.title("Offsets error (units) vs Absolute Error (CFO EST) (units)")
    plt.xlabel("Offsets error (units)")
    plt.ylabel("Absolute Error (CFO EST) (units)")
    plt.savefig("outputs_ftdomain/abs_error_c3_{}.png".format(datetime.datetime.now()))

    plt.figure(figsize=(12,10))
    plt.plot(m_tdx_arr, mse_error_c1, "MSE ERR MLE")
    plt.plot(m_tdx_arr, mse_error_c2, "MSE ERR RMCE")
    plt.title("Timing error (units) vs Mean Squared Error (CFO EST) (units)")
    plt.xlabel("Timing error (units)")
    plt.ylabel("Mean Squared Error (CFO EST) (units)")
    plt.savefig("outputs_ftdomain/mse_error_c12_{}.png".format(datetime.datetime.now()))

    plt.figure(figsize=(12,10))
    plt.plot(offs_arr, mse_error_c3, "MSE ERR SF PARK")
    plt.title("Offsets error (units) vs Mean Squared Error (CFO EST) (units)")
    plt.xlabel("Offsets error (units)")
    plt.ylabel("Mean Squared Error (CFO EST) (units)")
    plt.savefig("outputs_ftdomain/mse_error_c3_{}.png".format(datetime.datetime.now()))

    plt.figure(figsize=(12,10))
    plt.plot(m_tdx_arr, 100*mpe_error_c1, "MPE ERR MLE")
    plt.plot(m_tdx_arr, 100*mpe_error_c2, "MPE ERR RMCE")
    plt.title("Timing error (units) vs Mean Percentage Error (CFO EST) (%)")
    plt.xlabel("Timing error (units)")
    plt.ylabel("Mean Percentage Error (CFO EST) (%)")
    plt.savefig("outputs_ftdomain/mpe_error_c12_{}.png".format(datetime.datetime.now()))

    plt.figure(figsize=(12,10))
    plt.plot(offs_arr, 100*mpe_error_c3, "MPE ERR SF PARK")
    plt.title("Offsets error (units) vs Mean Percentage Error (CFO EST) (units)")
    plt.xlabel("Offsets error (units)")
    plt.ylabel("Mean Percentage Error (CFO EST) (units)")
    plt.savefig("outputs_ftdomain/mpe_error_c3_{}.png".format(datetime.datetime.now()))

    fig=plt.figure(figsize=(12,10))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(oo2d, mm2d, full_abs_error_c4, alpha=0.5)
    ax.set_xlabel("Offset error (units)")
    ax.set_ylabel("M Value (int)")
    ax.set_zlabel("Absolute Error (units)")
    plt.title("Offset Error (units) and M value to Absolute Error ")
    plt.savefig("outputs_ftdomain/abs_error_c4.png")

    fig=plt.figure(figsize=(12,10))
    ax = plt.axes() 
    plt.pcolormesh(oo2d, mm2d, full_abs_error_c4, shading="gouraud", cmap="rainbow")
    ax.contour(oo2d, mm2d, full_abs_error_c4, cmap="rainbow")
    ax.set_xlabel("Offset error (units)")
    ax.set_ylabel("M Value (int)")
    plt.title("Offset Error (units) and M value to Absolute Error ")
    color_map = cm.ScalarMappable(cmap="rainbow")
    color_map.set_array(full_abs_error_c4.flatten())
    cbar=fig.colorbar(color_map) # Add a colorbar to a plot
    cbar.set_label(' Absolute Error (units)', labelpad=5, fontsize=10)
    plt.grid(True)
    plt.savefig("results/abs_error_c4_cont.png")

    fig=plt.figure(figsize=(12,10))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(oo2d, mm2d, full_mse_error_c4, alpha=0.5)
    ax.set_xlabel("Offset error (units)")
    ax.set_ylabel("M Value (int)")
    ax.set_zlabel("Mean Sq. Error (units)")
    plt.title("Offset Error (units) and M value to Mean Sq. Error ")
    plt.savefig("outputs_ftdomain/mse_error_c4.png")

    fig=plt.figure(figsize=(12,10))
    ax = plt.axes() 
    plt.pcolormesh(oo2d, mm2d, full_mse_error_c4, shading="gouraud", cmap="rainbow")
    ax.contour(oo2d, mm2d, full_mse_error_c4, cmap="rainbow")
    ax.set_xlabel("Offset error (units)")
    ax.set_ylabel("M Value (int)")
    plt.title("Offset Error (units) and M value to Mean Sq. Error ")
    color_map = cm.ScalarMappable(cmap="rainbow")
    color_map.set_array(full_mse_error_c4.flatten())
    cbar=fig.colorbar(color_map) # Add a colorbar to a plot
    cbar.set_label(' Mean Sq. Error (units)', labelpad=5, fontsize=10)
    plt.grid(True)
    plt.savefig("results/mse_error_c4_cont.png")

    fig=plt.figure(figsize=(12,10))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(oo2d, mm2d, full_mpe_error_c4, alpha=0.5)
    ax.set_xlabel("Offset error (units)")
    ax.set_ylabel("M Value (int)")
    ax.set_zlabel("Mean Perc. Error (units)")
    plt.title("Offset Error (units) and M value to Mean P. Error ")
    plt.savefig("outputs_ftdomain/mpe_error_c4.png")

    fig=plt.figure(figsize=(12,10))
    ax = plt.axes() 
    plt.pcolormesh(oo2d, mm2d, full_mpe_error_c4, shading="gouraud", cmap="rainbow")
    ax.contour(oo2d, mm2d, full_mpe_error_c4, cmap="rainbow")
    ax.set_xlabel("Offset error (units)")
    ax.set_ylabel("M Value (int)")
    plt.title("Offset Error (units) and M value to Mean Perc. Error ")
    color_map = cm.ScalarMappable(cmap="rainbow")
    color_map.set_array(full_mpe_error_c4.flatten())
    cbar=fig.colorbar(color_map) # Add a colorbar to a plot
    cbar.set_label(' Mean Perc. Error (units)', labelpad=5, fontsize=10)
    plt.grid(True)
    plt.savefig("results/mpe_error_c4_cont.png")

else:
    
    print("Invalid sequence. Terminated.")

#end
