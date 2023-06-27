import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from smb_unzip.smb_unzip import smb_unzip
import datetime
from pathlib import Path
import argparse; parser = argparse.ArgumentParser(description='RAW_DATA folder path')

f_s = 10**6
f_c = 172*10**6
burst_period = 20*10**3
calm_period = 1*10**6
mean_amp = np.sqrt(2)
mean_val_noise = np.sqrt(2*np.pi)

def waterfall_visualization(data_dir: Path, FFT_LEN = 2048, f_s = 2000000, fft_bin = 564, f_c = 172500000):

    raw_files = sorted(glob.glob(os.path.join(data_dir, 'RAW_DATA_*')))
	samples = []

	for raw_file in raw_files:
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
		plt.savefig('ping_signal.png')
		plt.close()


def stft(samples: list, FFT_LEN: int) -> tuple:  # raw viz by latest method (june 2023)
    '''
        Perform STFT(Short Time Fourier Transform) on the sampled signal. The freq. range depends on the range given by np.fft.fftfreq(), which is [-f_s/2,f_s/2]. 
        return the time array, freq. array and the 2D power spectrum over the time-freq. plain 
    '''

    seq_samples = np.array(samples)
    arr_samples = np.reshape(seq_samples[0:FFT_LEN*int(
        len(samples) / FFT_LEN)], (FFT_LEN, int(len(samples) / FFT_LEN)), order='F')
    f_samp = np.fft.fft(arr_samples, axis=0) / FFT_LEN

    power = np.power(np.abs(np.fft.fftshift(f_samp, axes=0)), 2)
    freq = np.arange(-f_s/2, f_s/2, f_s/FFT_LEN)
    t = np.arange(int(len(seq_samples)/FFT_LEN)) * FFT_LEN / f_s

    return t, freq, power


def generate_3d_visualization(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    '''
        function for 3D visualization. Check https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html for tutorial and examples
    '''

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('time(s)', labelpad=20)
    ax.set_ylabel('freq(Hz)', labelpad=20)
    ax.set_zlabel('power', labelpad=20)

    x_grid, y_grid = np.meshgrid(x, y)
    surf = ax.plot_surface(x_grid, y_grid, z, cmap=plt.cm.cividis)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.savefig('JFTplot_CombinedRawFiles_{}.jpg'.format(
        datetime.datetime.now()))
    plt.show()


def combine_RAW_signal(RAW_signal_collection: list) -> list:
    '''
        Combine the RAW_DATA signals in the following way to simulate tower-sampled signal:
        for each RAW signal
            1. impose burst-calm period
            2. add random time offset
            3. add random amplitude and noise
        and finally sum all the signals up to form the returned combined signal
    '''

    collection_len = len(RAW_signal_collection)
    signal_len = len(RAW_signal_collection[0])

    burst_period_arr = np.random.uniform(0.9, 1.1, collection_len)
    calm_period_arr = np.random.uniform(0.9, 1.1, collection_len)

    calm_period_j = np.round(calm_period*calm_period_arr)
    burst_period_j = np.round(burst_period*burst_period_arr)

    random_offset_period_arr = np.random.uniform(0, 1.1, collection_len)
    random_offset_period_j = np.round(random_offset_period_arr*burst_period)

    amplitude_arr = np.random.uniform(
        0.9*mean_amp, 1.1*mean_amp, collection_len)
    noise_i = np.random.uniform(-1*mean_val_noise, 1*mean_val_noise, (collection_len, signal_len))

    combined_signal = np.zeros(signal_len, dtype=np.complex128)

    for i in range(len(RAW_signal_collection)):
        signal = RAW_signal_collection[i]
        # 0 out the inactive periods
        num_cycle = int(np.ceil(len(signal)/(burst_period_j[i]+calm_period_j[i])))
        for k in range(num_cycle):
            signal[int(k*calm_period_j[i]+k*burst_period_j[i]):int((k+1)*(calm_period_j[i])+k*(burst_period_j[i]))] = 0
        # time shift
        signal = np.roll(signal, int(random_offset_period_j[i]))
        signal[:int(random_offset_period_j[i])] = 0
        # add amplitude and noise
        combined_signal += amplitude_arr[i]*signal+noise_i[i]
    return combined_signal


# rawdata read and transform to signal
def read_single_RAW(dataFilePath: Path) -> list:
    '''
        read a single RAW_DATA file
    '''

    print(f'reading {dataFilePath}...')
    nSamples = int(dataFilePath.stat().st_size / 4)
    signal_raw = np.zeros(nSamples, dtype=np.complex128)
    with open(dataFilePath, 'rb') as dataFile:
        for i in range(nSamples):
            sampleBytes = dataFile.read(4)
            re, im = struct.unpack("<2h", sampleBytes)
            signal_raw[i] = float(re) / 0x7fff + float(im) * 1j / 0x7fff
    # t_raw = np.arange(0, nSamples / f_s, 1/f_s)
    print("Signal Generated.")
    return signal_raw


def read_raw_files(RAW_DATA_path: Path, countLimit: int) -> list[list]:
    '''
        read countLimit number of RAW_DATA files from the specified RAW_DATA_path
    '''

    RAW_signal_collection = []
    count = 0
    for child in RAW_DATA_path.iterdir():
        if child.parts[-1].startswith('RAW_DATA_') and count < countLimit:
            count += 1
            RAW_signal_collection.append(read_single_RAW(child))
    return RAW_signal_collection


if __name__ == '__main__':
    parser.add_argument(
        '--path', help='path of RAW_DATA folder', required=True, type=Path)
    parser.add_argument(
        '--count', help='number of RAW_DATA files to read', required=True, type=int)
    parser.add_argument(
        '--fftlen', help='length of stft slices', required=True, type=int)

    args = parser.parse_args()

    RAW_signal_collection = read_raw_files(args.path, args.count)
    preprocessed_signal = combine_RAW_signal(RAW_signal_collection)
    t, freq, power = stft(samples=preprocessed_signal, FFT_LEN=args.fftlen)
    generate_3d_visualization(t, freq, power)
