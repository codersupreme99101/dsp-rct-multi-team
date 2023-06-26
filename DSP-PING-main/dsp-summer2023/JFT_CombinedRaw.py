import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from smb_unzip.smb_unzip import smb_unzip
import glob
import datetime
from pathlib import Path

f_s=10**6 #hyperparams
f_c=172*10**6
burst_period=20*10**3
calm_period=1*10**6#40*10**4 #2x burst for demo purposes #1*10**6 #real value
mean_amp=np.sqrt(2) #units 
mean_val_noise=np.sqrt(2*np.pi)

def generate_visualization(samples): #raw viz by latest method (june 2023)
    
    FFT_LEN = 2048
    # fft_bin = 564
            
    seq_samples = np.array(samples)
    arr_samples = np.reshape(seq_samples[0:FFT_LEN*int(len(samples) / FFT_LEN)], (FFT_LEN, int(len(samples) / FFT_LEN)), order='F')
    f_samp = np.fft.fft(arr_samples, axis=0) / FFT_LEN
    waterfall = np.power(np.abs(np.fft.fftshift(f_samp, axes=0)), 2)

    # waterfall_extents = ((f_c - f_s / 2) / 1e6, (f_c + f_s / 2) / 1e6, len(seq_samples) / f_s, 0)

    # freq_isolated_power = waterfall[fft_bin,:]
    freq=np.arange(-f_s, f_s, 2*f_s/FFT_LEN)
    t = np.arange(int(len(seq_samples)/FFT_LEN)) * FFT_LEN / f_s
   
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')
    
    t_grid, freq_grid = np.meshgrid(t, freq)
    surf = ax.plot(t_grid, freq_grid, waterfall, cmap = plt.cm.cividis)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    # Set axes label
    ax.set_xlabel('time(s)', labelpad=20)
    ax.set_ylabel('freq(MHz)', labelpad=20)
    ax.set_zlabel('power', labelpad=20)
    ax.set_zlim(0,0.008)
    plt.savefig('JFTplot_CombinedRawFiles_{}.jpg'.format(datetime.datetime.now()))
    plt.show()
    return t,freq,waterfall

def make_signal(t_signal_raw):
    # t_signal_raw consists of (t_raw,signal_raw)
    burst_period_arr=np.random.uniform(0.9, 1.1,len(t_signal_raw)) #+/- 10% of the given mean data at norm. 1
    calm_period_arr=np.random.uniform(0.9, 1.1, len(t_signal_raw)) #multiplied, integer due to each number being the 1us sample 

    calm_period_j=np.round(calm_period*calm_period_arr) #In sample space, at least 1 in f_s 
    burst_period_j=np.round(burst_period*burst_period_arr) #int for random value to integer

    random_offset_period_arr=np.random.uniform(0,1.1, len(t_signal_raw)) #non to 110% usual
    random_offset_period_j=np.round(random_offset_period_arr*burst_period) #some multiple of the burst period, for some simulated realistic values 
    
    amplitude_arr=np.random.uniform(0.9*mean_amp, 1.1*mean_amp, len(t_signal_raw))
    noise_i=np.random.uniform(-1*mean_val_noise, 1*mean_val_noise, (len(t_signal_raw),len(t_signal_raw[0][1])))
    
    combined_signal=np.zeros(len(t_signal_raw[0][1]), dtype=np.complex128)
    for i in range(len(t_signal_raw)):
        t=t_signal_raw[i][0]
        signal=t_signal_raw[i][1]
        # 0 out the inactive periods
        num_cycle=int(np.ceil(len(t)/(burst_period_j[i]+calm_period_j[i])))
        for k in range(num_cycle):
            signal[int(k*calm_period_j[i]+k*burst_period_j[i]):int((k+1)*(calm_period_j[i])+k*(burst_period_j[i]))]=0
        # time shift
        signal=np.roll(signal, int(random_offset_period_j[i]))
        signal[:int(random_offset_period_j[i])]=0 
        # add amplitude and noise
        combined_signal+=amplitude_arr[i]*signal+noise_i[i]
    return combined_signal

def generate_real_signal(dataFilePath): #rawdata read and transform to signal 
        nSamples = int(os.path.getsize(dataFilePath) / 4)
        signal_raw = np.zeros(nSamples, dtype=np.complex128)
        with open(dataFilePath, 'rb') as dataFile:
            for i in range(nSamples):
                sampleBytes = dataFile.read(4)
                re, im = struct.unpack("<2h", sampleBytes)
                signal_raw[i] = float(re) / 0x7fff + float(im) * 1j / 0x7fff
        t_raw = np.arange(0, nSamples / f_s, 1/f_s)
        print("Real Signal Generated.")
        return t_raw, signal_raw
    
def read_raw_files(RAW_DATA_path,countLimit):
    data_collection=[]
    count=0
    for root, dirs, files in os.walk(RAW_DATA_path):
        for file in files:
            if file.startswith('RAW_DATA_') and count<countLimit:
                count+=1
                path=os.path.join(root,file)
                print(f'reading {path}...')
                t_signal_raw=generate_real_signal(path)
                data_collection.append(t_signal_raw)
    return data_collection



if __name__ == '__main__':
    t_signal_raw=read_raw_files('/home/haw057/RCT/RCT_Haochen/generate_signal_4testing/2017.07.06.RCT_Test/2017.07.06.RCT_Test/RUN_000001',1) #limit num of files read
    # print(len(t_signal_raw[0][1])) checks time length of signal - about 17 seconds
    preprocessed_signal=make_signal(t_signal_raw)
    generate_visualization(samples=preprocessed_signal)