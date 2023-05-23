import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct
import os

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
    data_collection=[]
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.startswith('RAW_DATA_'):
                path=os.path.join(root,file)
                print(f'reading {path}...')
                t_signal_raw=generate_real_signal(path,1000000)
                data_collection.append(t_signal_raw)
    return data_collection

res=read_all_raw_files()   
