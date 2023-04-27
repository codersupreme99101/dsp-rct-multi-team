from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
import warnings 
from scipy import signal
from scipy.signal.filter_design import freqs_zpk
import scipy

plt.style.use('seaborn-poster')

option=2 #0,1, 2, 3

t_end = 4 # s
f_s = 1000000 # Hz #t_s for psi
f_c = 172000000 # Hz
f_t_arr = np.array([5000])#, 25000])#np.array([50, 2550, 5050, 7550, 10050, 12550, 15050, 17550, 20050, 22550, 25050, 27550, 30050, 32550, 35050]) # Hz offset from center to guess
t_ping = 0.05 # s
ping_period = 1 # s
ping_power = -96 # dB
noise_power = -60 # dB

f_t_arr=f_t_arr[:5]

print(f_t_arr)
print(f_t_arr[0])

min_channel_separation=2500 #hz, offset difference of multisources wrt each other at min. 

f_ping=1 #Hz
t_bin_ping=1/f_ping

golden_ping_idx_ov = []

ping_signal_ov=[]

def generate_test_signal(): # Computed signal parameters

        ping_amplitude = 10 ** (ping_power / 20) # FS
        ping_length = int(t_ping * f_s) # samples
        ping_time_index = np.arange(0, ping_length)

        ping_signal=np.zeros(ping_time_index.shape, dtype=np.complex128)

        for i in range(len(f_t_arr)): #multi
            ping_signal += np.cos(f_t_arr[i] / 2 * np.pi* ping_time_index) + 1j * np.sin(f_t_arr[i] / 2 * np.pi* ping_time_index)

        ac_sig=ping_signal
        ac_sig=np.array(ac_sig)

        golden_ping_idx = []

        noise_snr = 10.0**(noise_power/10.0)
        ping_wait_signal = np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(int((ping_period - t_ping) * f_s), 2)).view(np.complex128) # Generate noise with calculated power #data signal , length=N
        ping_signal_noise = ping_amplitude * ping_signal + np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len(ping_signal), 2)).view(np.complex128).reshape(len(ping_signal)) # Generate noise with calculated power #pilot signal, length=M

        signal_test = np.array([0.00001])
        for i in range(int(t_end / ping_period)):
            signal_test = np.append(signal_test, ping_signal_noise) #appended Pilot block at Pi
            golden_ping_idx.append(len(signal_test))
            signal_test = np.append(signal_test, ping_wait_signal) #appended data block at Di
        signal_test = np.append(signal_test, ping_signal_noise) #from this #Y[k]

        signal_magnitude = np.abs(signal_test)
        signal_power = 20 * np.log10(signal_magnitude)
        t_test = np.arange(len(signal_power)) / f_s

        golden_ping_idx_ov=np.array(golden_ping_idx)

        golden_ping_idx_ov=golden_ping_idx_ov/f_s

        ping_signal_ov=np.array(ping_signal)

        print(golden_ping_idx_ov)

        pso=ping_signal_ov

        return t_test, signal_test, pso, golden_ping_idx_ov

def autocorrelate(signal, pso):

    warnings.filterwarnings("ignore")

    AC_window_len = 0.001 # s
    lag = 0.001 # s

    AC_window_len_s = 2048
    template = np.flip(pso[:AC_window_len_s])
    AC_f_s = f_s / 1024

    # AC_signal = np.correlate(template, signal, mode='valid')
    AC_signal = np.zeros(int(len(signal) / 1024), dtype=complex)
    for i in range(int((len(signal) - AC_window_len_s) / 1024)):
    #     AC_signal[i] = np.max(np.correlate(ping_signal[0:0 + AC_window_len_s], signal[i + lag_s:i + lag_s + AC_window_len_s], "same"))
        AC_signal[i] = np.sum(np.multiply(template, signal[i * 1024 : i * 1024 + AC_window_len_s]))

    AC_signal_magnitude = np.abs(AC_signal)
    AC_signal_power = 20 * np.log10(AC_signal_magnitude)
    t = np.arange(len(AC_signal_power)) / AC_f_s

    return t, AC_signal_power

def get_ping_rssi(AC_signal):

    AC_f_s = f_s / 1024

    ping_len_AC_s = int(t_ping * AC_f_s)
    ping_q_len_AC_s = int((ping_period - t_ping) * AC_f_s)
    ping_template_AC = np.concatenate((np.ones(ping_len_AC_s, dtype=complex), np.ones(ping_q_len_AC_s, dtype=complex) * 1e-9))

    ping_ideal = np.convolve(ping_template_AC, AC_signal, mode='valid')
    ping_rssi = 20 * np.log10(np.abs(ping_ideal))
    t = np.arange(len(ping_rssi)) / AC_f_s

    return t, ping_rssi

def locate_power(ping_rssi):

    AC_f_s = f_s / 1024

    ping_locator = ping_rssi > 3
    t = np.arange(len(ping_locator)) / AC_f_s

    ping_t = []
    for i in range(len(ping_locator) - 1):
        if ping_locator[i] == 0 and ping_locator[i + 1] == 1:
            ping_t.append((i + 1) / AC_f_s)

    ping_t=np.array(ping_t)
    print("Pings at {}".format(ping_t))

    return t, ping_t

def perform_prematch(signal, pso, gpi,  name):

    plt.figure(figsize=(12,10))

    t_ac, ac_sig_pow=autocorrelate(signal, pso)

    plt.plot(t_ac, ac_sig_pow, color="blue")
    t_ping, ping_rssi=get_ping_rssi(ac_sig_pow)

    f1=-ping_rssi+np.min(ping_rssi)

    plt.plot(t_ping, f1, color="purple")

    plt.savefig("outputs/all_analysis_pings_{}.png".format(name))

    smart_ping_find(t_ping, f1, signal, gpi)

    pass

def smart_ping_find(times_rssi, rssi_signal, signal, gpi):

    new_rssi_sig=-rssi_signal-np.min(-rssi_signal)

    #method to detect top n values, where n is determined from max arrangement of signal by value output, and a statistical function
    #and using this value to see in the n peaks, which 2 peaks are less than avg of x distance between other peak values, and avg them.
    #store x occurences of this peak and then that is output (final)
    # statstical function to use is: some function that detects the vast differences between nearest neighbors, like dirac delta points in local 1D domain
    #known ping frequencies, nominally, apriori, use those to sample at integer intervals of 1/f_ping, and get a max point per sample length, 
    # for the full sample length

    pings_found=[]

    num_breaks=int(np.ceil(np.max(times_rssi)/t_bin_ping))

    len_pblock=int(len(times_rssi)/num_breaks)

    broken_times=[]
    broken_rssi=[]

    print(times_rssi.shape)
    print(rssi_signal.shape)

    for i in range(num_breaks):

        if (i+1)*t_bin_ping>len(times_rssi):
            stopper=len(times_rssi)
        else:
            stopper=(i+1)*len_pblock
        broken_times.append(times_rssi[i*len_pblock:stopper])
        broken_rssi.append(new_rssi_sig[i*len_pblock:stopper])

    broken_times=np.array(broken_times)
    broken_rssi=np.array(broken_rssi)

    print(broken_rssi)

    plt.figure(figsize=(12,10))

    print(broken_times.shape)
    print(broken_rssi.shape)

    color=["red", "blue", "green", "orange", "purple", "black"]

    max_vals=[]

    for i in range(num_breaks):

        plt.plot(broken_times[i], broken_rssi[i], color=color[i])
        max_vals.append(np.min(broken_rssi[i]))

    max_vals=np.array(max_vals)

    true_pings=[]

    for i in range(len(gpi)):
        
        true_pings.append(8.5)

    true_pings=np.array(true_pings)

    plt.scatter(gpi, true_pings, color="black")

    for i in range(len(max_vals)):
        mvi=max_vals[i]
        ival=np.argwhere(broken_rssi[i]==mvi)[0][0]
        tval=broken_times[i][ival]
        pings_found.append(tval)

    pings_found=np.array(pings_found)

    print(pings_found)

    decode_p=[]

    for i in range(len(pings_found)):

        decode_p.append(5)

    decode_p=np.array(decode_p)

    cleaned_ping_values=get_cleaned_pings(pings_found)

    decode_cp=[]

    for i in range(len(cleaned_ping_values)):

        decode_cp.append(7.5)

    decode_cp=np.array(decode_cp)

    plt.scatter(pings_found, decode_p, color="purple")

    plt.scatter(cleaned_ping_values, decode_cp, color="cyan")

    print(cleaned_ping_values)

    mse=0

    for i in range(len(cleaned_ping_values)):
        mse+=(cleaned_ping_values[i]-pings_found[i])**2

    mse=mse/len(cleaned_ping_values)

    print(mse) #in s

    t_e, f_e=waterfall_analysis(signal)
    
    plt.plot(t_e, f_e*10**7, color="black")
    
    plt.grid(True)

    plt.savefig("outputs/split_pings.png")

    pass

def get_cleaned_pings(pf):

    t_bin_ping=1 #given already

    offset=np.abs((np.std(pf)/len(pf))*0.112) #usually an underestimate, but now very minimal overestimate 

    print(offset)

    cpv=[]

    first_val=pf[0]+offset

    for i in range(len(pf)):
        if i==0:
            cpv.append(first_val)
        else:
            cpv.append(cpv[0]+t_bin_ping*i)

    cpv=np.array(cpv)

    return cpv

def waterfall_analysis(samples):

    FFT_LEN = 2048
    f_s = 1000000
    f_c = 172000000

    target_freq = f_s+f_c

    integral_len = int(np.floor(0.06 * f_s / FFT_LEN))
    freqs = np.fft.fftshift(np.fft.fftfreq(FFT_LEN, 1.0/f_s))
    closest_freq = min(freqs, key=lambda x:abs(x - (target_freq - f_c)))
    fft_bin = min(range(len(freqs)), key=lambda x: abs(freqs[x] - (target_freq - f_c)))

    seq_samples = np.array(samples)
    arr_samples = np.reshape(seq_samples[0:FFT_LEN*int(len(samples) / FFT_LEN)], (FFT_LEN, int(len(samples) / FFT_LEN)), order='F')
    f_samp = np.fft.fft(arr_samples, axis=0) / FFT_LEN
    waterfall = np.power(np.abs(np.fft.fftshift(f_samp, axes=0)), 2)

    waterfall_extents = ((f_c - f_s / 2) / 1e6, (f_c + f_s / 2) / 1e6, len(seq_samples) / f_s, 0)

    freq_isolate = waterfall[fft_bin,:]
    freq_energy = np.zeros(int(len(freq_isolate) / integral_len))

    for i in range(len(freq_energy)):
        freq_energy[i] = np.sum(freq_isolate[i * integral_len:(i + 1) * integral_len - 1])
    t = np.arange(len(freq_energy)) / (len(freq_energy) - 1) * waterfall_extents[2]

    """

    plt.figure(figsize=(12,10))

    plt.plot(t, freq_energy)
    plt.ylabel('Energy')
    plt.xlabel('Time (s)')
    plt.title('Ping Signal Energy')

    plt.savefig("outputs/energy_pings.png")

    plt.figure(figsize=(12,10))

    plt.plot(waterfall[fft_bin,:])
    plt.ylabel('Signal Strength (dBFS)')
    plt.xlabel('Sample')
    plt.title("Ping Power")

    plt.savefig("outputs/power_pings.png")


    """

    return t, freq_energy


t_test, signal_test, pso, gpi=generate_test_signal()


t=t_test

n_src= 2 #len(f_t_arr) #2 # min

x=signal_test

X = fft(x)

def plot_fft_quick(X):

    plt.figure(figsize=(12,10))

    X=signal.spectrogram((np.real(X)), scaling="density") #density or spectrum #spectrogram #real, image, magn 

    print(X)
    print(len(X[0]))

    plt.savefig("outputs/fft_quick.png")

    pass

#plot_fft_quick(x)

N = len(X) 
n = np.arange(N)
T = N/f_c
freq = n/T 

x_a=np.abs(X)

ix_a=ifft(x_a)

#perform_prematch(ix_a, pso, gpi, "ifft")

def noise_db_experiment():
    
    ping_power_main = -96 # dB
    noise_power_main = -60 # dB
    
    n_testpts=100 #for now
    
    mse_arr=[]
    noise_power_arr=np.linspace(-100,0,n_testpts)
    
    ping_power=ping_power_main
    
    for noise_power in noise_power_arr:
    
        t_test, signal_test, pso, gpi=generate_test_signal()
        
        x=signal_test

        X = fft(x)
                
        N = len(X) 
        n = np.arange(N)
        T = N/f_c
        freq = n/T 

        x_a=np.abs(X)

        ix_a=ifft(x_a)
        
        signal=ix_a
        
        t_ac, ac_sig_pow=autocorrelate(signal, pso)
        t_ping, ping_rssi=get_ping_rssi(ac_sig_pow)
        f1=-ping_rssi+np.min(ping_rssi)
        
        times_rssi=t_ping
        rssi_signal=f1
        new_rssi_sig=-rssi_signal-np.min(-rssi_signal)
        
        pings_found=[]

        num_breaks=int(np.ceil(np.max(times_rssi)/t_bin_ping))

        len_pblock=int(len(times_rssi)/num_breaks)

        broken_times=[]
        broken_rssi=[]
        
        for i in range(num_breaks):
        
            if (i+1)*t_bin_ping>len(times_rssi):
                stopper=len(times_rssi)
            else:
                stopper=(i+1)*len_pblock
            broken_times.append(times_rssi[i*len_pblock:stopper])
            broken_rssi.append(new_rssi_sig[i*len_pblock:stopper])

        broken_times=np.array(broken_times)
        broken_rssi=np.array(broken_rssi)
        
        max_vals=[]

        for i in range(num_breaks):

            max_vals.append(np.min(broken_rssi[i]))

        max_vals=np.array(max_vals)

        true_pings=[]

        for i in range(len(gpi)):
            
            true_pings.append(8.5)

        true_pings=np.array(true_pings)

        for i in range(len(max_vals)):
            mvi=max_vals[i]
            ival=np.argwhere(broken_rssi[i]==mvi)[0][0]
            tval=broken_times[i][ival]
            pings_found.append(tval)

        pings_found=np.array(pings_found)

        cleaned_ping_values=get_cleaned_pings(pings_found)
        
        mse=0

        for i in range(len(cleaned_ping_values)):
            mse+=(cleaned_ping_values[i]-pings_found[i])**2

        mse=mse/len(cleaned_ping_values)

        mse_arr.append(mse)
    
    mse_arr=np.array(mse_arr)
    
    plt.figure(figsize=(12,10))

    plt.plot(noise_power_arr, mse_arr, color="red")
    plt.ylabel('MSE for Noise (un-normalized)')
    plt.xlabel('Noise Power (dB)')
    plt.title('Noise Analysis')
    plt.grid(True)

    plt.savefig("outputs/noise_db_chk.png")
    
    plt.figure(figsize=(12,10))

    plt.plot(noise_power_arr/ping_power_main, mse_arr, color="green")
    plt.ylabel('MSE for Noise (normalized)')
    plt.xlabel('Noise Power / Ping Power')
    plt.title('Noise Analysis')
    plt.grid(True)

    plt.savefig("outputs/noise_db_chk_norm.png")
    
    pass
    
#noise_db_experiment()

def power_db_experiment():
    
    ping_power_main = -96 # dB
    noise_power_main = -60 # dB
    
    n_testpts=100 #for now
    
    mse_arr=[]
    power_arr=np.linspace(-100,0,n_testpts)
    
    noise_power=noise_power_main
    
    for ping_power in power_arr:
    
        t_test, signal_test, pso, gpi=generate_test_signal()
        
        x=signal_test

        X = fft(x)
        
        signal=ifft(signal_test)
        
        N = len(X) 
        n = np.arange(N)
        T = N/f_c
        freq = n/T 

        x_a=np.abs(X)

        ix_a=ifft(x_a)
        
        signal=ix_a
        
        t_ac, ac_sig_pow=autocorrelate(signal, pso)
        t_ping, ping_rssi=get_ping_rssi(ac_sig_pow)
        f1=-ping_rssi+np.min(ping_rssi)
        
        times_rssi=t_ping
        rssi_signal=f1
        new_rssi_sig=-rssi_signal-np.min(-rssi_signal)
        
        pings_found=[]

        num_breaks=int(np.ceil(np.max(times_rssi)/t_bin_ping))

        len_pblock=int(len(times_rssi)/num_breaks)

        broken_times=[]
        broken_rssi=[]
        
        for i in range(num_breaks):
        
            if (i+1)*t_bin_ping>len(times_rssi):
                stopper=len(times_rssi)
            else:
                stopper=(i+1)*len_pblock
            broken_times.append(times_rssi[i*len_pblock:stopper])
            broken_rssi.append(new_rssi_sig[i*len_pblock:stopper])

        broken_times=np.array(broken_times)
        broken_rssi=np.array(broken_rssi)
        
        max_vals=[]

        for i in range(num_breaks):

            max_vals.append(np.min(broken_rssi[i]))

        max_vals=np.array(max_vals)

        true_pings=[]

        for i in range(len(gpi)):
            
            true_pings.append(8.5)

        true_pings=np.array(true_pings)

        for i in range(len(max_vals)):
            mvi=max_vals[i]
            ival=np.argwhere(broken_rssi[i]==mvi)[0][0]
            tval=broken_times[i][ival]
            pings_found.append(tval)

        pings_found=np.array(pings_found)

        cleaned_ping_values=get_cleaned_pings(pings_found)
        
        mse=0

        for i in range(len(cleaned_ping_values)):
            mse+=(cleaned_ping_values[i]-pings_found[i])**2

        mse=mse/len(cleaned_ping_values)

        mse_arr.append(mse)
    
    mse_arr=np.array(mse_arr)
    
    plt.figure(figsize=(12,10))

    plt.plot(power_arr, mse_arr, color="red")
    plt.ylabel('MSE for Pings (un-normalized)')
    plt.xlabel('Pings Power (dB)')
    plt.title('Ping Analysis')
    plt.grid(True)

    plt.savefig("outputs/pings_db_chk.png")
    
    plt.figure(figsize=(12,10))

    plt.plot(noise_power_main/power_arr, mse_arr, color="green")
    plt.ylabel('MSE for Pings (normalized)')
    plt.xlabel('Noise Power / Ping Power')
    plt.title('Ping Analysis')
    plt.grid(True)

    plt.savefig("outputs/pings_db_chk_norm.png")
    
    pass

#power_db_experiment()

def func1_convert(x):
    
    return 10**(x/20)

def func2_convert(x):
    
    return 10**(x/10)

def power_single_method():
    
    ping_power_main = -96 # dB
    noise_power_main = -60 # dB
    
    n_testpts=100 #for now
    
    mse_arr=[]
    ping_power_arr=np.linspace(-100,0,n_testpts)
    
    noise_power=noise_power_main
    
    for ping_power in ping_power_arr:
    
        t_test, signal_test, pso, gpi=generate_test_signal()
        
        signal=signal_test
        
        t_ac, ac_sig_pow=autocorrelate(signal, pso)
        t_ping, ping_rssi=get_ping_rssi(ac_sig_pow)
        
        time_ping, cleaned_ping_values=locate_power(ping_rssi)
        
        pings_found=gpi
        
        mse=0

        for i in range(len(cleaned_ping_values)):
            mse+=(cleaned_ping_values[i]-pings_found[i])**2

        mse=mse/len(cleaned_ping_values)

        mse_arr.append(mse)
    
    mse_arr=np.array(mse_arr)
    
    plt.figure(figsize=(12,10))

    plt.plot(ping_power_arr, mse_arr, color="red")
    plt.ylabel('MSE for Ping - Single (un-normalized)')
    plt.xlabel('Ping Power - Single (dB)')
    plt.title('Ping Analysis - Single ')
    plt.grid(True)

    plt.savefig("outputs/pings_db_chk_single.png")
    
    plt.figure(figsize=(12,10))

    plt.plot(noise_power_main/ping_power_arr, mse_arr, color="green")
    plt.ylabel('MSE for Pings - Single (normalized)')
    plt.xlabel('Noise Power / Ping Power - Single')
    plt.title('Ping Analysis - Single')
    plt.grid(True)

    plt.savefig("outputs/pings_db_chk_norm_single.png")
    
    pass

def noise_single_method():
    
    ping_power_main = -96 # dB
    noise_power_main = -60 # dB
    
    n_testpts=100 #for now
    
    mse_arr=[]
    noise_power_arr=np.linspace(-100,0,n_testpts)
    
    ping_power=ping_power_main
    
    for noise_power in noise_power_arr:
    
        t_test, signal_test, pso, gpi=generate_test_signal()
        
        signal=signal_test
        
        t_ac, ac_sig_pow=autocorrelate(signal, pso)
        t_ping, ping_rssi=get_ping_rssi(ac_sig_pow)
        
        time_ping, cleaned_ping_values=locate_power(ping_rssi)
        
        pings_found=gpi
        
        mse=0

        for i in range(len(cleaned_ping_values)):
            mse+=(cleaned_ping_values[i]-pings_found[i])**2

        mse=mse/len(cleaned_ping_values)

        mse_arr.append(mse)
    
    mse_arr=np.array(mse_arr)
    
    plt.figure(figsize=(12,10))

    plt.plot(noise_power_arr, mse_arr, color="red")
    plt.ylabel('MSE for Noise - Single (un-normalized)')
    plt.xlabel('Noise Power - Single (dB)')
    plt.title('Noise Analysis - Single ')
    plt.grid(True)

    plt.savefig("outputs/noise_db_chk_single.png")
    
    plt.figure(figsize=(12,10))

    plt.plot(noise_power_arr/ping_power_main, mse_arr, color="green")
    plt.ylabel('MSE for Noise - Single (normalized)')
    plt.xlabel('Noise Power / Ping Power - Single')
    plt.title('Noise Analysis - Single')
    plt.grid(True)

    plt.savefig("outputs/noise_db_chk_norm_single.png")
    
    pass

#noise_single_method()
#power_single_method()

full_x=len(x)
ratio_cut=0.1
len_cut=int(ratio_cut*full_x)
x=x[:len_cut]

def costas_loop_order4(samples):

    N = len(samples)
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.005
    beta = 0.001
    out = np.zeros(N, dtype=np.complex)
    freq_log = []
    for i in range(N):
        out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
        error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        freq_log.append(freq * f_s / (2*np.pi)) # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

    # Plot freq over time to see how long it takes to hit the right offset
    plt.figure(figsize=(12,10))
    plt.plot(freq_log,'.-')
    plt.xlabel("sample")
    plt.ylabel("offset")
    plt.title("CFO Costas - Order 4")
    plt.savefig("outputs/costas_out.png")

    pass

#costas_loop_order4(x)

def cfo_alpha_off(signal, a):
    
    alpha_off=0
    
    b=a #a=16 or 64
    n=b
    
    sum_comp=0
    for i in range(n):
        conj_sig=np.real(signal[i])-1j*np.imag(signal[i])
        sig=np.real(signal[i+a])+1j*np.imag(signal[i+a])
        sum_comp+=(conj_sig*sig)
        
    alpha_off_i=sum_comp/b
    
    alpha_off=np.arctan(np.imag(alpha_off_i)/np.real(alpha_off_i))
    
    return alpha_off

c64=cfo_alpha_off(x, 64)
print(c64) #trials 
c16=cfo_alpha_off(x, 16)
print(c16)

print(func1_convert(c16))
print(func2_convert(c16))
print(func1_convert(c64))
print(func1_convert(c64))

t_c=1/(360*f_c) #1 degree phase change interval
t_c_r=1/(2*np.pi*f_c)

print(1/(t_c*c64))
print(1/(t_c_r*c64))

print(1/(t_c*c16))
print(1/(t_c_r*c16))

def cfo_blue_off(signal):
    
    t_max=len(x)/f_s
    min_fr=20 #hz
    
    r=int(t_max)*f_s
    u_large=int(t_max/(1/min_fr))
    
    n_large=u_large*r
    k_large=int(u_large/2)
    
    last_pb=0
    sum_k=0
    for u in range(0,k_large+1):
        sum_complements=0
        for m in range(u*r,n_large):
            if m<len(signal):
                orig=np.real(signal[m])+1j*np.imag(signal[m])
                comp=np.real(signal[m-u*r])-1j*np.imag(signal[m-u*r])
                sum_complements+=(orig*comp)
        
        phi_blue=sum_complements/(n_large-u*r)
        print(phi_blue, last_pb)
        if np.real(phi_blue)==0 or np.real(last_pb)==0:
            phi_u=np.pi/2
        else:
            phi_u=((np.arctan(np.imag(phi_blue)/np.real(phi_blue)))-(np.arctan(np.imag(last_pb)/np.real(last_pb))))%(2*np.pi)
        if phi_blue==0:
            break
        last_pb=phi_blue
        w_u=((3*u_large-u)*((u_large-u+1)-(k_large*(u_large-k_large))))/(k_large*(4*k_large**2-6*u_large*k_large+3*u_large**2-1))
        if u!=1:
            sum_k+=(w_u*phi_u)
    
    cfo_offset_b=(sum_k*f_s*u_large)/(2*np.pi)

    return cfo_offset_b

#print(cfo_blue_off(x))

def cfo_mle_off(signal):
    
    t_max=len(x)/f_s
    large_r=int(f_s*t_max)
    fac=1/(2*np.pi*len(signal)*(1/f_s))
    
    sum_r=0
    min_fr=20 #hz
    u=int(t_max/(1/min_fr))
    m=large_r*u
    for r in range(0, large_r):
        val=np.real(signal[m-r])+1j*np.imag(signal[m-r])
        comp=np.real(signal[m-r-len(x)])-1j*np.imag(signal[m-r-len(x)])
        sum_r+=(val*comp)
    
    return fac*np.arctan(np.imag(sum_r)/np.real(sum_r))

#print(cfo_mle_off(x))

#whats a good n for n of cfos present??? 2,5,15 at most 

#approach for exact ping detection: 

#NOTES:

"""
START: (predicted ping offsets are form current system involving IFFT)

1. for a good range of trials of numbers of transmitters, and the offset array, run it and see predictions to actual pings
2. see that for the actual to the predicted, get std. dev. or mean, or another statistical measure
3. offset by that measure, see if the accuracies closely match or not, if not:
3a. try a mathematical model fit (function for describing offset in values, with array of offsets as a parameter and its length (number of N))
3b. if that fails, stick to a decen statistical measure, or
4. conclude that the best method of ping detection is reached. 

END: (Done)
"""

def calc_parabola_vertex(freqs, values):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''
    x1 = freqs[0]
    x2 = freqs[1]
    x3 = freqs[2]
    y1 = values[0]
    y2 = values[1]
    y3 = values[2]

    denom = np.float64((x1-x2) * (x1-x3) * (x2-x3))
    a = np.float64((x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2))) / denom
    b = np.float64((x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3))) / denom
    c = np.float64((x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3)) / denom

    return a, b, c

def interpolation(freqs, values):
    """
    used for interpolating results from periodogram
    :param freqs: Top 3 frequencies
    :param values: Top 3 values for the frequencies
    :return: interpolated frequency
    """

    a, b, c = calc_parabola_vertex(freqs, values)
    return -b/(2.0*a)

def CramerRaoBound(n=16, wmin=0.2, wmax=0.25, smin=-20, smax=40):
    snrs_db = np.linspace(smin, smax, num=4*(smax-smin))
    snrs = 10**(snrs_db/10.0)
    crb1 = 7.36/(np.sqrt(snrs)*n**3)
    crb2 = 9.42/(snrs*n**3)
    crb = np.maximum(crb1, crb2)
    factor = (wmax-wmin)/np.pi
    crb = factor**2*crb
    crb = 10*np.log10(crb)
    return snrs_db, crb

def Periodogram_estimator(data):
    """
    Implements the MLE error for frequency estimation, which is the periodogram method
    :param data: input time domain data
    :param fs: sampling frequency
    :return: highest frequency peak of the periodogram
    """
    length = 2**15
    dfft = np.absolute(fft(np.complex128(data), n=length), dtype=np.float128)
    frequencies = np.float128(np.fft.fftfreq(length))
    interp = False

    # interpolate from top 3 frequencies
    idx = np.argmax(np.abs(dfft))
    
    if interp is True:
        if idx == length-1:
            mle_freq = frequencies[idx]

        else:
            indices = [idx-1, idx, idx+1]
            freqs = frequencies[indices].astype(np.float64)
            values = np.abs(dfft[indices]).astype(np.float64)

            mle_freq = interpolation(freqs, values)
    else:
        return frequencies[idx]
        
    return mle_freq

def Welchs_Rect_Periodogram(data):
    """
    Implements Welch's method of periodogram with rectangular window
    :param data: input time domain data
    :param fs: sampling frequency
    :return: highest frequency peak of the periodogram
    """
    length = len(data) #2**15
    frequencies, dfft = np.absolute(scipy.signal.welch(data, nfft=length, window='boxcar', nperseg=len(data)//2), dtype=np.float128)
    interp = False

    # interpolate from top 3 frequencies
    idx = np.argmax(np.abs(dfft))
    
    if interp is True:
        if idx == length-1:
            mle_freq = frequencies[idx]

        else:
            indices = [idx-1, idx, idx+1]
            freqs = frequencies[indices].astype(np.float64)
            values = np.abs(dfft[indices]).astype(np.float64)

            mle_freq = interpolation(freqs, values)
    else:
        return frequencies[idx]
        
    return mle_freq

def WPA_estimator_1(data):
    """
    Estimates the frequency of an input data set using the weighted phase averager
    :param data: Input time domain data
    :return: The highest estimated digital frequency
    """
    data = data.astype(np.complex128)
    n = np.float64(len(data))
    w_t = np.array([], dtype=np.float64)
    w_hat = np.float64(0.0)
    data_conj = np.copy(np.conj(data)).astype(np.complex128)
    data_conj = data_conj[:-1]
    data = data[1:].astype(np.complex128)  # remove the first value so that it will be data_conj[t]*data[t+1]

    for t in range(int(n)-1):
        w_t = np.append(w_t, ((3.0 / 2.0) * n / (n ** 2.0 - 1.0) * (1.0 - ((np.float64(t) - (n / 2.0 - 1.0)) / (n / 2.0)) ** 2.0)))

    w_hat = np.float64(np.sum(np.multiply(w_t, np.angle(np.multiply(data_conj, data))))/(2.0*np.pi).real)
    # print(w_hat)
    return w_hat

#test 3 new methods:

#"""
full_x=len(x)
ratio_cut=0.2
len_cut=int(ratio_cut*full_x)
x=x[:len_cut]

warnings.filterwarnings("ignore")

periodogram_w=Periodogram_estimator(x)
welch_w=Welchs_Rect_Periodogram(x)
wpa_w=WPA_estimator_1(x)

print(periodogram_w, welch_w, wpa_w)

f_wpa=welch_w
f_per=periodogram_w
f_wel=welch_w #assigned 

print(func1_convert(f_wpa), func1_convert(f_per), func1_convert(f_wel))
print(func2_convert(f_wpa), func2_convert(f_per), func2_convert(f_wel))

period_deg=57.2958*periodogram_w
welch_deg=57.2958*welch_w
wpa_deg=57.2958*wpa_w
t_c=1/(360*f_c) #1 degree phase change interval

f_wpa=1/(t_c*wpa_deg)
f_per=1/(t_c*period_deg)
f_wel=1/(t_c*welch_deg)

print(f_wpa, f_per, f_wel)

print(func1_convert(f_wpa), func1_convert(f_per), func1_convert(f_wel))
print(func2_convert(f_wpa), func2_convert(f_per), func2_convert(f_wel))

f_wpa=1/(t_c*wpa_w)
f_per=1/(t_c*periodogram_w)
f_wel=1/(t_c*welch_w)

print(f_wpa, f_per, f_wel)

print(func1_convert(f_wpa), func1_convert(f_per), func1_convert(f_wel))
print(func2_convert(f_wpa), func2_convert(f_per), func2_convert(f_wel))

t_c_r=1/(2*np.pi*f_c) #1 degree phase change interval, rads

f_wpa=1/(t_c_r*wpa_w)
f_per=1/(t_c_r*periodogram_w)
f_wel=1/(t_c_r*welch_w)

print(f_wpa, f_per, f_wel)

print(func1_convert(f_wpa), func1_convert(f_per), func1_convert(f_wel))
print(func2_convert(f_wpa), func2_convert(f_per), func2_convert(f_wel))

#"""

