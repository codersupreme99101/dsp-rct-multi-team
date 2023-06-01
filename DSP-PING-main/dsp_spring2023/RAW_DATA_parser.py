import numpy as np
import struct
import os
from pathlib import Path
from smb_unzip.smb_unzip import smb_unzip #be sure to follow instructions: https://github.com/UCSD-E4E/smb-unzip 


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
