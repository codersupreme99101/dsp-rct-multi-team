import numpy as np
import struct
import os
from smb_unzip.smb_unzip import smb_unzip #be sure to follow instructions: https://github.com/UCSD-E4E/smb-unzip 
import matplotlib.pyplot as plt; plt.ion()
import glob
import datetime
from pathlib import Path

def generate_raw_visualization(self, f_s, f_c): #raw viz by latest method (june 2023)

    FFT_LEN = 2048
    fft_bin = 564
            
    data_dir =smb_unzip(network_path='smb://nas.e4e.ucsd.edu/rct/data/set_1/',output_path=Path('.'),username='aryakeni',password='****') #not actual password
    raw_files = sorted(glob.glob(os.path.join(data_dir, 'RAW_DATA_*')))

    samples = []

    for raw_file in raw_files[7:8]:
        with open(raw_file, 'rb') as data_file:
            data = data_file.read()
        for i in range(int(len(data) / 4)):
            tsample = struct.unpack('hh', data[i * 4:(i+1) * 4])
            sample = (tsample[0] + tsample[1] * 1j) / 1024
            samples.append(sample)
        seq_samples = np.array(samples)
        arr_samples = np.reshape(seq_samples[0:FFT_LEN*int(len(samples) / FFT_LEN)], (FFT_LEN, int(len(samples) / FFT_LEN)), order='F')
        f_samp = np.fft.fft(arr_samples, axis=0) / FFT_LEN
        waterfall = np.power(np.abs(np.fft.fftshift(f_samp, axes=0)), 2)

        waterfall_extents = ((f_c - f_s / 2) / 1e6, (f_c + f_s / 2) / 1e6, len(seq_samples) / f_s, 0)

        freq_isolate = waterfall[fft_bin,:]

        t = np.arange(len(freq_isolate)) / (len(freq_isolate) - 1) * waterfall_extents[2]
        fig1 = plt.figure()
        plt.plot(t, freq_isolate)
        plt.ylabel('Power')
        plt.xlabel('Time (s)')
        plt.title('Ping Signal')
        plt.savefig('ping_signal_{}.png'.format(datetime.datetime.now()))
        plt.close()
                
    return waterfall, freq_isolate

def generate_real_signal(dataFilePath,f_s): #rawdata read and transform to signal 
    
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
    
def read_all_raw_files():
    dataset_subfolders=["set_1", "set_2", "set_3", "set_4"] #for folders under data, containing RAW only. 
    data_collection=[]
    for folder in dataset_subfolders:
        midpath="smb://nas.e4e.ucsd.edu/rct/data/{}/".format(folder)
        for filename in os.listdir(midpath):
            path=os.path.join(midpath, filename)
            if os.path.isfile(path) and "RAW_DATA_" in path:
                print(f'reading {path}...')
                t_signal_raw=generate_real_signal(path,1000000)
                data_collection.append(t_signal_raw)
    return data_collection

res=read_all_raw_files()   
