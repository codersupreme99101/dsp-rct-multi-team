from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')

option=2 #0,1, 2, 3

t_end = 4 # s
f_s = 1000000 # Hz #t_s for psi
f_c = 172000000 # Hz
f_t_arr = np.array([5000, 6000, 7000]) # Hz offset from center to guess
t_ping = 0.05 # s
ping_period = 1 # s
ping_power = -96 # dB
noise_power = -60 # dB

def generate_test_signal(): # Computed signal parameters

        ping_amplitude = 10 ** (ping_power / 20) # FS
        ping_length = int(t_ping * f_s) # samples
        ping_time_index = np.arange(0, ping_length)

        ping_signal=np.zeros(ping_time_index.shape, dtype=np.complex128)

        for i in range(len(f_t_arr)): #multi
            ping_signal += np.cos(f_t_arr[i] / 2 * np.pi* ping_time_index) + 1j * np.sin(f_t_arr[i] / 2 * np.pi* ping_time_index)

        ac_sig=ping_signal
        ac_sig=np.array(ac_sig)

        noise_snr = 10.0**(noise_power/10.0)
        ping_wait_signal = np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(int((ping_period - t_ping) * f_s), 2)).view(np.complex128) # Generate noise with calculated power #data signal , length=N
        ping_signal_noise = ping_amplitude * ping_signal + np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len(ping_signal), 2)).view(np.complex128).reshape(len(ping_signal)) # Generate noise with calculated power #pilot signal, length=M

        n=len(ping_wait_signal)
        m=len(ping_signal_noise)
        b=n
        p=m
        q=m+n
        b_dash=b+1

        signal_test = np.array([0.00001])
        golden_ping_idx = []
        for i in range(int(t_end / ping_period)):
            signal_test = np.append(signal_test, ping_signal_noise) #appended Pilot block at Pi
            golden_ping_idx.append(len(signal_test))
            signal_test = np.append(signal_test, ping_wait_signal) #appended data block at Di
        signal_test = np.append(signal_test, ping_signal_noise) #from this #Y[k]

        signal_magnitude = np.abs(signal_test)
        signal_power = 20 * np.log10(signal_magnitude)
        t_test = np.arange(len(signal_power)) / f_s

        return t_test, signal_test

t_test, signal_test=generate_test_signal()

if option==0: #positive simple 

    f_c=10**6 #function works excellent for a given range, known limits of freq from 0 to f_c/2 

    # sampling rate
    sr = f_c #change to max possible frequency (which is given f_c (central freq) / 2)
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,100,ts)

    freq_arr=np.array([99000, 88000, 77000, 110000, 110200])

    n_src= 2 #len(freq_arr)

    x=np.zeros(t.shape, dtype=np.complex128)

    for i in range(len(freq_arr)):

        x += 3*(np.cos(2*np.pi*freq_arr[i]*t)+1j*np.sin(2*np.pi*freq_arr[i]*t))

    x=np.array(x)

    x_clean=x #clean signal for reference, if needed 

    x += np.random.normal(500, np.sqrt(500), len(x))

    X = fft(x)
    N = len(X) 
    n = np.arange(N)
    T = N/sr
    freq = n/T 

    x_a=np.abs(X)

    f0=freq[:int(len(freq)/2)]
    x0=x_a[:int(len(freq)/2)]

    f1=f0[1:]
    x1=x0[1:]

    #plt.plot(f1, x1)
    #plt.savefig("test_ece_fft_p.png")

    xs=np.sort(x1)[::-1]
    xs1=xs[:n_src]

    #print(xs1) #magnitudes 

    f_keys=[]
    for i in range(len(xs1)):
        i_v=np.argwhere(xs1[i]==x_a)[0][0]
        f_keys.append(freq[i_v])

    f_keys=np.array(f_keys)

    print(f_keys)

elif option==1: #-ve simple 

    f_c=10**6 #flip logic (same but the arange if same -f_c/2 to 0), from 0 option 

    # sampling rate
    sr = f_c #change to max possible frequency (which is given f_c (central freq) / 2)
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,100,ts)

    #freq_arr=np.array([-99000, -88000, -77000, -110000, -110200])

    freq_arr=np.array([99000, 88000, 77000, 110000, 110200])

    n_src= 2 #len(freq_arr)

    x=np.zeros(t.shape, dtype=np.complex128)

    for i in range(len(freq_arr)):

        x += 3*(np.cos(2*np.pi*freq_arr[i]*t)+1j*np.sin(2*np.pi*freq_arr[i]*t))

    x=np.array(x)

    x_clean=x #clean signal for reference, if needed 

    x += np.random.normal(500, np.sqrt(500), len(x))

    X = fft(x)
    N = len(X) 
    n = np.arange(N)
    T = N/sr
    freq = n/T 

    freq_neg=-freq

    x_a=np.abs(X)

    f0=freq_neg[:int(len(freq_neg)/2)]
    x0=x_a[:int(len(freq_neg)/2)]

    f1=f0[1:]
    x1=x0[1:]

    #plt.plot(f1, x1)
    #plt.savefig("test_ece_fft_n.png")

    xs=np.sort(x1)[::-1]
    xs1=xs[:n_src]

    #print(xs1) #magnitudes 

    f_keys=[]
    for i in range(len(xs1)):
        i_v=np.argwhere(xs1[i]==x_a)[0][0]
        f_keys.append(freq[i_v])

    f_keys=-1*np.array(f_keys)

    print(f_keys)

elif option==2: #actual test +ve

    t=t_test

    n_src= 2 #len(f_t_arr) #2 # min

    x=signal_test

    X = fft(x)
    N = len(X) 
    n = np.arange(N)
    T = N/f_c
    freq = n/T 

    x_a=np.abs(X)

    f0=freq[:int(len(freq)/2)]
    x0=x_a[:int(len(freq)/2)]

    f1=f0[1:]
    x1=x0[1:]

    #plt.plot(f1, x1)
    #plt.savefig("test_ece_fft_n.png")

    xs=np.sort(x1)[::-1]
    xs1=xs[:n_src]

    #print(xs1) #magnitudes 

    f_keys=[]
    for i in range(len(xs1)):
        i_v=np.argwhere(xs1[i]==x_a)[0][0]
        f_keys.append(freq[i_v])

    f_keys=np.array(f_keys)

    absv=np.abs(f_keys[0]-f_keys[1]) # 1 metric to do it, can see if std. dev is more than certain threshold 
    if absv<=200: #hz
        print("1 Source only")
    else:
        print("2+ sources")

    print(f_keys)

elif option==3: #actual test -ve 
    
    t=t_test

    n_src= 2 #len(f_t_arr) #2 # min 

    x=signal_test

    X = fft(x)
    N = len(X) 
    n = np.arange(N)
    T = N/f_c
    freq = n/T 

    freq_n=-freq

    x_a=np.abs(X)

    f0=freq_n[:int(len(freq_n)/2)]
    x0=x_a[:int(len(freq_n)/2)]

    f1=f0[1:]
    x1=x0[1:]

    #plt.plot(f1, x1)
    #plt.savefig("test_ece_fft_n.png")

    xs=np.sort(x1)[::-1]
    xs1=xs[:n_src]

    #print(xs1) #magnitudes 

    f_keys=[]
    for i in range(len(xs1)):
        i_v=np.argwhere(xs1[i]==x_a)[0][0]
        f_keys.append(freq[i_v])

    f_keys=-1*np.array(f_keys)

    absv=np.abs(f_keys[0]-f_keys[1]) # 1 metric to do it, can see if std. dev is more than certain threshold 
    if absv<=200: #hz
        print("1 Source only")
    else:
        print("2+ sources")

    print(f_keys)

#PLAN:

#absultely horrible fft spectral accuracy, but detects n>1 sources anyways, which is crucial 
#use to find if the diff betw the 2 values is more than 200 Hz, then yes, 2 srcs truly exist

