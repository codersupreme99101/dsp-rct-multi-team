from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
import warnings 
from scipy import signal
from scipy.signal.filter_design import freqs_zpk

plt.style.use('seaborn-poster')

option=2 #0,1, 2, 3

t_end = 4 # s
f_s = 1000000 # Hz #t_s for psi
f_c = 172000000 # Hz
f_t_arr = np.array([50, 2550, 5050, 7550, 10050, 12550, 15050, 17550, 20050, 22550, 25050, 27550, 30050, 32550, 35050]) # Hz offset from center to guess #np.array([1000, 2000, 3000, 5000, 7000, 10000, 15000, 15550, 20000]) #
t_ping = 0.05 # s
ping_period = 1 # s
ping_power = -96 # dB
noise_power = -60 # dB

min_channel_separation=2500 #hz, offset difference of multisources wrt each other at min. 

#write subroutine to check if each channel is separated by the minimum value or not 

def adjust_min_sep(channel_links): #ascending order array 

    min_chan_arr_adj=[]

    min_df=2*f_c+1 

    for i in range(len(channel_links)-1):
        j=i+1
        diff_ij=channel_links[j]-channel_links[i]
        if diff_ij<min_df:
            min_df=diff_ij

    enlarged_fac=np.abs(min_channel_separation/min_df)

    min_chan_arr_adj=np.array(channel_links*enlarged_fac)

    return min_chan_arr_adj

#method 2

def adjust_min_sep_v2(channel_links): #ascending order array 

    count_min_violated=1

    while count_min_violated!=0:
        count_min_violated=0
        for i in range(len(channel_links)):
            for j in range(len(channel_links)):
                if i!=j:
                    if np.abs(channel_links[i]-channel_links[j])<min_channel_separation:
                        count_min_violated+=1
                        min_diff=np.abs(min_channel_separation-np.abs(channel_links[i]-channel_links[j]))
                        if channel_links[i]>channel_links[j]:
                            channel_links[i]+=min_diff/2
                            channel_links[j]-=min_diff/2
                        else:
                            channel_links[i]-=min_diff/2
                            channel_links[j]+=min_diff/2
                        count_min_violated-=1

    return channel_links

#f_t_arr=adjust_min_sep(f_t_arr) #adjust_min_sep_v2(f_t_arr)

f_ping=1 #Hz
t_bin_ping=1/f_ping

golden_ping_idx_ov = []

ping_signal_ov=[]

#assuming pre-intra-channel separation is at least the minimum value 
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

        #code for 20ms chunking of frequencies 

        chunker=20*10**-3 #20ms #40*10**-3 #40ms #20-40ms allowed 

        len_signal=len(signal_test)
        time_chunk=int(chunker/(1/f_s))

        wait_amp=1 #2 #1-2 in sec. 
        time_wait=int(wait_amp*(1/(1/f_s))) #1-2 seconds 

        ping_wait_signal_chunk=np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len_signal,))+1j*np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len_signal,))

        ping_noise_chunk = ping_amplitude * (np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len_signal,))+1j*np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len_signal,))) + ping_wait_signal_chunk
        
        chunked_signal=np.zeros((len_signal,), dtype=np.complex128)

        max_timer=int(np.floor(len_signal/(time_chunk+time_wait)))+1
        counter_freqs=0

        idx_adder=0

        for i in range(max_timer):

            if counter_freqs>=len(f_t_arr):
                counter_freqs=0

            ##code for wait time by time_wait

            if idx_adder>len_signal:
                arr_ch=np.linspace(idx_adder, len_signal, int(len_signal-idx_adder))
                chunked_signal[idx_adder:len_signal]= ping_amplitude* np.cos(f_t_arr[counter_freqs] / 2 * np.pi* arr_ch) + 1j * np.sin(f_t_arr[counter_freqs] / 2 * np.pi* arr_ch)

            else:
                arr_ch=np.linspace(idx_adder, idx_adder+time_chunk, int(time_chunk))
                chunked_signal[idx_adder:idx_adder+time_chunk]= ping_amplitude* np.cos(f_t_arr[counter_freqs] / 2 * np.pi* arr_ch) + 1j * np.sin(f_t_arr[counter_freqs] / 2 * np.pi* arr_ch)

            idx_adder+=time_chunk
            idx_adder+=time_wait

            counter_freqs+=1

        signal_test=np.array(chunked_signal)+ping_noise_chunk

        t_test=np.arange(len(signal_test))

        return t_test, signal_test, pso, golden_ping_idx_ov

def plot_raw_signal(x, y):

    plt.figure(figsize=(12,10))

    plt.plot(x, np.real(y), color="purple", label="real")
    plt.plot(x, np.imag(y), color="green", label="imag")
    plt.plot(x, (np.imag(y)**2+np.real(y)**2)**0.5, color="red", label="pwr")

    plt.title("Signal Recording in Timeseries")
    plt.xlabel("Timesteps (per {}s)".format((1/f_s)))
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(loc="best")

    plt.savefig("results/chopped_sig_overall.png")

    pass

#"""
if __name__=="__main__":
    
    t_x, s_y, _, _ = generate_test_signal()
    chop=100 #len(s_y)
    t_x=t_x[:chop]
    s_y=s_y[:chop] #predisposition 
    plot_raw_signal(t_x, s_y)
#"""

#end