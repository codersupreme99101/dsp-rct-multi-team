import numpy as np
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
import warnings
import os
import datetime
import time
from sklearn import decomposition
from pathlib import Path
from smb_unzip.smb_unzip import smb_unzip #be sure to follow instructions: https://github.com/UCSD-E4E/smb-unzip 

class CFO_DSP:

    #INIT PARAMS: 

    def __init__(self): # Generated signal parameters

        self.t_end = 4 # s
        self.f_s = 1000000 # Hz #t_s for psi
        self.f_c = 172000000 # Hz
        self.f_t = 5000 # Hz offset from center to guess
        self.t_ping = 0.05 # s
        self.ping_period = 1 # s
        self.ping_power = -96 # dB
        self.noise_power = -60 # dB
        self.dataFilePath=None #real load 
        self.ac_sig=[] #param ints to load into when fns call 
        self.p=0#pilot block indexes 
        self.n=-1 #data length
        self.m=-1 #pilot length
        self.b=-1 #generalized n 
        self.b_dash=-1 #nonreuse
        self.q=-1 #trans block 
        self.psi=1024 #decimation
        self.d=4 #recall, ignored
        self.wu=[2,4,6,8,16] #w factor for Q,B hardset 
        self.types_arr=["AOM-NR","MOA-NR","AOM-R","MOA-R", "conventional", "MLE", "BLUE"]
        self.mse_arr=[] #arrs for analysis fills 
        self.std_arr=[]
        self.crb_arr=[]
        self.offsets=[]
        self.cpe=[]
        self.evm=[]
        self.threshold=[]
        self.corrected_sig_dict={0:[],1:[],2:[],3:[],4:[],5:[],6:[]} #dict for cpe fills per method 
        self.signal_type_print="" #signal type 
        self.precompute_bq=False #B,Q Overwrite
        self.hardset_hyper=False #hyperparam hardsets 
        self.noise_applied="" #type of noise applied
        self.sig_normalized=False #normalization? 
        self.utype_sample_evm="" #type of evm sample 
        self.min_mse_type=""
        self.min_mse_val=-1
        self.min_std_type=""
        self.min_std_val=-1
        self.min_crb_type=""
        self.min_crb_val=-1
        self.min_cpe_type=""
        self.min_cpe_val=-1
        self.min_evm_type=""
        self.min_evm_val=-1
        self.max_threshold_val=-1
        self.max_threshold_type=""
        self.min_mse_offset=-1
        self.min_crb_offset=-1
        self.min_evm_offset=-1
        self.min_cpe_offset=-1
        self.min_std_offset=-1
        self.max_threshold_offset=-1
        self.maxfilenum=-1
        self.experiment_type=""
        self.sample_mse_arr=[] #nested array of per sample values of metric arrays 
        self.sample_std_arr=[]
        self.sample_crb_arr=[]
        self.sample_offsets=[]
        self.sample_cpe=[]
        self.sample_evm=[]
        self.sample_threshold=[]
        self.threshold_value=200 #Hz
        self.offset_n_trans=[] #offsets for n transmitters 
        self.sample_vals=[]
        self.offset_vals=[]
        self.mse_i=[] #temp plotters for sample for each metric 
        self.crb_i=[]
        self.tso_i=[]
        self.evm_i=[]
        self.cpe_i=[]
        self.std_i=[]
        self.mse_ii=[] #global storers of analysis ""
        self.crb_ii=[]
        self.tso_ii=[]
        self.evm_ii=[]
        self.cpe_ii=[]
        self.std_ii=[]
        self.mse_j=[]#temp plotters for offset for each metric 
        self.crb_j=[]
        self.tso_j=[]
        self.evm_j=[]
        self.cpe_j=[]
        self.std_j=[]
        self.mse_jj=[]#global storers of analysis ""
        self.crb_jj=[]
        self.tso_jj=[]
        self.evm_jj=[]
        self.cpe_jj=[]
        self.std_jj=[]
        self.min_sample_crb=-1 #sample extrema from analysis
        self.min_sample_cpe=-1
        self.max_sample_tso=-1
        self.min_sample_std=-1
        self.min_sample_mse=-1
        self.min_sample_evm=-1
        self.min_offset_sample_crb=-1 #sample extra of offsets from analysis 
        self.min_offset_sample_evm=-1
        self.min_offset_sample_cpe=-1
        self.min_offset_sample_mse=-1
        self.max_offset_sample_tso=-1
        self.min_offset_sample_std=-1
        self.s_evm=-1 #offset value at sample pt of extrema 
        self.s_cpe=-1
        self.s_std=-1
        self.s_mse=-1
        self.s_tso=-1
        self.s_crb=-1
        self.multi_signal=[]
        self.nested_signals=[]
        self.super_mse_arr=[] #nested array of per offset values of metric arrays 
        self.super_std_arr=[]
        self.super_crb_arr=[]
        self.super_cpe_arr=[]
        self.super_evm_arr=[]
        self.super_tso_arr=[]
        self.min_vv_crb=-1 #offset extrema from analysis
        self.min_vv_cpe=-1
        self.max_vv_tso=-1
        self.min_vv_std=-1
        self.min_vv_mse=-1
        self.min_vv_evm=-1
        self.svv_evm=-1 #offset extrema from offset analysis 
        self.svv_cpe=-1
        self.svv_std=-1
        self.svv_mse=-1
        self.svv_tso=-1
        self.svv_crb=-1 
        self.n_trans=-1 #number of transm for decomp. 
        self.mixing_coeffs=[]
        self.inter_mc=[]

    #OS mgmt:

    def detectdir(self): #makes dir in spec. folder
    
        path = "results/"
        super_counter = 0
        for filename in os.listdir(path):
            f = os.path.join(path, filename)
            if f.startswith(path+"trial_"):
                super_counter += 1
        main_name = "trial_{}_{}".format(self.stype, super_counter+1)
        path_main = os.path.join(path, main_name)
        os.mkdir(path_main)

        print("\nDirectory {} created in the sub-folder of 'results/'. \n".format(main_name))

        pass 

    def do_getmaxnum(self): #gets current maximum trial postfix num 
    
        path = "results/"
        super_counter = 0
        for filename in os.listdir(path):
            f = os.path.join(path, filename)
            if f.startswith(path + "trial_"):
                super_counter += 1

        return super_counter

    #Generated signal data: 

    def generate_test_signal(self): # Computed signal parameters

        ping_amplitude = 10 ** (self.ping_power / 20) # FS
        ping_length = int(self.t_ping * self.f_s) # samples
        ping_time_index = np.arange(0, ping_length)
        ping_signal = np.cos(self.f_t / 2 * np.pi* ping_time_index) + 1j * np.sin(self.f_t / 2 * np.pi* ping_time_index)

        self.ac_sig=ping_signal
        self.ac_sig=np.array(self.ac_sig)

        noise_snr = 10.0**(self.noise_power/10.0)
        ping_wait_signal = np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(int((self.ping_period - self.t_ping) * self.f_s), 2)).view(np.complex128) # Generate noise with calculated power #data signal , length=N
        ping_signal_noise = ping_amplitude * ping_signal + np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len(ping_signal), 2)).view(np.complex128).reshape(len(ping_signal)) # Generate noise with calculated power #pilot signal, length=M

        self.n=len(ping_wait_signal)
        self.m=len(ping_signal_noise)
        self.b=self.n
        self.p=self.m
        self.q=self.m+self.n
        self.b_dash=self.b+1

        signal_test = np.array([0.00001])
        golden_ping_idx = []
        for i in range(int(self.t_end / self.ping_period)):
            signal_test = np.append(signal_test, ping_signal_noise) #appended Pilot block at Pi
            golden_ping_idx.append(len(signal_test))
            signal_test = np.append(signal_test, ping_wait_signal) #appended data block at Di
        signal_test = np.append(signal_test, ping_signal_noise) #from this #Y[k]

        signal_magnitude = np.abs(signal_test)
        signal_power = 20 * np.log10(signal_magnitude)
        t_test = np.arange(len(signal_power)) / self.f_s

        print("\nTest Signal Generated. \n")

        return t_test, signal_test

    def generate_real_signal(self): #rawdata read and transform to signal 

        self.dataFilePath=smb_unzip(network_path='smb://nas.e4e.ucsd.edu/rct/data/set_1/RAW_DATA_000001_000002',output_path=Path('.'),username='aryakeni',password='****') #not actual password
        nSamples = int(os.path.getsize(self.dataFilePath) / 4)
        signal_raw = np.zeros(nSamples, dtype=np.complex128)
        with open(self.dataFilePath, 'rb') as dataFile:
            for i in range(nSamples):
                sampleBytes = dataFile.read(4)
                re, im = struct.unpack("<2h", sampleBytes)
                signal_raw[i] = float(re) / 0x7fff + float(im) * 1j / 0x7fff
        t_raw = np.arange(0, nSamples / self.f_s, 1/self.f_s)

        print("\nReal Signal Generated. \n")

        return t_raw, signal_raw

    #Plots: 

    def plot_test_signal(self, t, s): #signal itself plotted

        signal_magnitude = np.abs(s)
        signal_power = 20 * np.log10(signal_magnitude)
        plt.plot(t, signal_power)
        plt.title("Raw Received Signal, P_ping=%d dB, P_noise=%d dB" % (self.ping_power, self.noise_power))
        plt.xlabel("Time (s)")
        plt.ylabel("Received Power (dB)")
        plt.savefig("results/trial_{}_{}/raw_signal_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))

        print("\nTest signal plotted, data saved in directory. \n")

        pass

    def plot_real_signal(self, t, s): #signal itself plotted 

        signal_magnitude = np.abs(s)
        signal_power = 20 * np.log10(signal_magnitude)
        mpl.rcParams['agg.path.chunksize'] = 1000000

        plt.plot(t[:len(t)-1], signal_power)
        plt.xlabel('Time (s)')
        plt.ylabel('Received Power (dB)')
        plt.title('Time Series Recieved Real Signal Plot at %s , P_ping=%d dB, P_noise=%d dB' % (self.dataFilePath, self.ping_power, self.noise_power), fontsize=6)
        plt.savefig("results/trial_{}_{}/real_signal_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))

        print("\nReal Signal plotted, data saved in directory. \n")

        pass

    #Transforms: 

    def normalize_signal(self, signal): #normalizes max power of signal to 1

        max_val=np.max(((np.real(signal)+np.imag(signal))**0.5)**2)
        norm_sig=signal/(max_val**0.5)

        print("\nSignal Normalized W.R.T. Power. \n")
        return norm_sig

    def autocorrelate_phi(self, signal, signal_shift): #ac for special Y to phi method in A/MOM/A N/R methods 

        return signal*(np.real(signal_shift)-np.imag(signal_shift))

    #CFO estimates:

    def cfo_est_blue(self, signal): #BLUE method

        total_n=500 #len(signal)
        u=int(self.m)
        r=int(total_n/u)
        k=int(u/2)
        fsum=0
        last_phiu=-1
        phiu=-1
        for i in range(k):
            mu=np.arange(i*r, total_n)
            mu2=mu-i*r
            a4=np.where(mu<len(signal),mu,mu-len(signal))
            a5=np.where(mu2<len(signal),mu2,mu2-len(signal))
            s1=np.take(signal, a4)
            s2=np.take(signal, a5)
            phi_bu=np.sum(self.autocorrelate_phi(s1, s2))/(total_n-i*r)
            if u>=1:
                phiu=(np.arctan(np.imag(phi_bu)/np.real(phi_bu))-np.arctan(np.imag(last_phiu)/np.real(last_phiu)))%(2*np.pi)
            else:
                phiu=phi_bu
            last_phiu=phiu
            wu=((3*(u-i)*(u-i+1))-(k*(u-k)))/(k*(4*k*k-6*u*k+3*u*u-1))
            fsum+=(wu*phiu)

        print("\nBLUE Method CFO Est. Done. \n")

        return 1/((fsum*u*self.f_s)/(2*np.pi))

    def cfo_est_aom_r(self,signal): #angle of mean with reuse 

        sum_aom_r=0
        k=np.arange(1,self.p+1)
        for i in range(1, self.b_dash+1):
            a1=self.q*i-self.q+k
            a2=self.q*i+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            s1=np.take(signal, a4)
            s2=np.take(signal, a5)
            sum_aom_r+=(np.sum(self.autocorrelate_phi(s1, s2)))
        ang_sig=sum_aom_r/(self.m*self.b)

        angle=np.arctan(np.imag(ang_sig)/np.real(ang_sig))

        print("\nAngle of Mean - Reuse method for CFO Est. Done. \n")

        return angle/(self.psi*self.q)

    def cfo_est_aom_nr(self, signal): #angle of mean with no reuse 
    
        sum_aom_nr=0
        k=np.arange(1,self.p+1)
        for i in range(1, self.b+1):
            a1=2*self.q*i-2*self.q+k
            a2=2*self.q*i-self.q+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal)) 
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            s1=np.take(signal, a6)
            s2=np.take(signal, a7) 
            sum_aom_nr+=(np.sum(self.autocorrelate_phi(s1,s2)))
        ang_sig=sum_aom_nr/(self.m*self.b)

        angle=np.arctan(np.imag(ang_sig)/np.real(ang_sig))

        print("\nAngle of Mean - NonReuse method for CFO Est. Done. \n")

        return angle/(self.psi*self.q)

    def cfo_est_moa_r(self, signal): #mean of angle with reuse 
    
        angle_sum=0
        k=np.arange(1,self.p+1)
        for i in range(1, self.b_dash+1):
            a1=self.q*i-self.q+k
            a2=self.q*i+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            s1=np.take(signal, a6)
            s2=np.take(signal, a7) 
            sum_moa_r=(np.sum(self.autocorrelate_phi(s1,s2)))/self.m
            angle=np.arctan(np.imag(sum_moa_r)/np.real(sum_moa_r))
            angle_sum+=angle 

        print("\nMean of Angle - Reuse method for CFO Est. Done. \n")

        return angle_sum/(self.psi*self.q*self.b)

    def cfo_est_moa_nr(self, signal): #mean of angle without reuse 
    
        angle_sum=0
        k=np.arange(1,self.p+1)
        for i in range(1, self.b+1):
            a1=2*self.q*i-2*self.q+k
            a2=2*self.q*i-self.q+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            s1=np.take(signal, a6)
            s2=np.take(signal, a7)
            sum_moa_nr=(np.sum(self.autocorrelate_phi(s1,s2)))/self.m
            angle=np.arctan(np.imag(sum_moa_nr)/np.real(sum_moa_nr))
            angle_sum+=angle 

        print("\nMean of Angle - NonReuse method for CFO Est. Done. \n")

        return angle_sum/(self.psi*self.q*self.b)

    def cfo_est_conventional(self, signal): #typical method for cfo 
    
        l_dash=int((len(signal)-1)/self.d)
        sum_phi=0
        i=np.arange(0,self.p-1)
        a1=self.q*i+l_dash
        a2=self.q*i+l_dash+self.q
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal))
        s1=np.take(signal, a4)
        s2=np.take(signal, a5)
        sum_phi=np.sum(self.autocorrelate_phi(s1,s2))

        print("\nConventional method for CFO Est. Done. \n")

        return (np.arctan((np.imag(sum_phi))/(self.m*(np.real(sum_phi)))))/(self.q*self.psi)

    def cfo_est_mle(self, signal): #MLE method

        l_dash=int((len(signal)-1)/self.d)
        w_mle=0
        for mu in range(1, self.b+1):
            sum_mle=0
            p=np.arange(mu, self.b+1)
            a1=l_dash+self.q*(p-mu)
            a2=l_dash+self.q*(p-mu)+mu*self.q 
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            s1=np.take(signal, a6)
            s2=np.take(signal, a7)
            sum_mle=np.sum(self.autocorrelate_phi(s1,s2))
            angle=np.arctan(np.imag(sum_mle)/np.real(sum_mle))
            w_mle+=(angle/mu)

        print("\nMLE method for CFO Est. Done. \n")

        return w_mle/(self.q*self.psi) 

    def crb_est(self, std): #cramer rao bound estimate 

        print("\nCramer-Rao Lower Bound for CFO Est. Done. \n")
    
        return ((6*std**2)*(std**2+self.b+1))/(self.m*(self.q**2)*(self.psi**2)*((self.b+1)**2)*((self.b+1)**2-1))

    def crb_blue_est(self, signal, std, ch): #crb est. for BLUE method

        if ch==0:

            print("\nCramer-Rao Lower Bound for BLUE Method CFO Est. Done. \n")

            return 3/(2*np.pi*np.pi*(10**((self.ping_power-self.noise_power)/10))*len(signal)*(1-(1/(len(signal)*len(signal)))))

        elif ch==1:

            print("\nCramer-Rao Lower Bound for CFO Est. Done. \n")
    
        return ((6*std**2)*(std**2+self.b+1))/(self.m*(self.q**2)*(self.psi**2)*((self.b+1)**2)*((self.b+1)**2-1))

    def cpe_est(self, signal, estimated_f_t, type): #estimates offset error within full signal

        l_dash=int((len(signal)-1)/self.d)
        signal_sum=0
        nop=-1
        if type=="nr":
            nop=self.p
        else:
            nop=self.p-1
        qu=np.arange(0,self.m)  

        for i in range(nop+1):
            a1=i*self.q+qu+l_dash
            a2=i*self.q+qu
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal)) 
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            s1=np.take(signal, a6)
            s2=np.take(signal, a7)
            signal_sum+=np.sum((np.exp(1j*estimated_f_t*self.psi*(a1))*s1*(np.real(s2)-np.imag(s2))))
            #print("Here-cpe-{}".format(i))

        print("\nCommon Phase Error (CPE) Estimation for CFO Est. Done. \n")
        
        return -np.arctan(np.imag(signal_sum)/np.real(signal_sum))

    def evm_est(self, signal, estimated_f_t, choice): #error vector magnitude in power form 

        signal_f_t=[]

        ping_amplitude = 10 ** (self.ping_power / 20) # FS
        ping_length = int(self.t_ping * self.f_s) # samples
        ping_time_index = np.arange(0, ping_length)
        ping_signal = np.cos(estimated_f_t / 2 * np.pi* ping_time_index) + 1j * np.sin(estimated_f_t / 2 * np.pi* ping_time_index)

        noise_snr = 10.0**(self.noise_power/10.0)
        ping_wait_signal = np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(int((self.ping_period - self.t_ping) * self.f_s), 2)).view(np.complex128) # Generate noise with calculated power #data signal , length=N
        ping_signal_noise = ping_amplitude * ping_signal + np.random.normal(0, np.sqrt(noise_snr*2.0)/2.0, size=(len(ping_signal), 2)).view(np.complex128).reshape(len(ping_signal)) # Generate noise with calculated power #pilot signal, length=M

        signal_test = np.array([0.00001])
        golden_ping_idx = []
        for i in range(int(self.t_end / self.ping_period)):
            signal_test = np.append(signal_test, ping_signal_noise) #appended Pilot block at Pi
            golden_ping_idx.append(len(signal_test))
            signal_test = np.append(signal_test, ping_wait_signal) #appended data block at Di
        signal_f_t = np.append(signal_test, ping_signal_noise) #from this #Y[k]

        signal_f_t=np.array(signal_f_t) 
        evm_sum=0
        if choice==0:
            u=len(signal)
        else: #1 made by program externally 
            u=500
            
        s1=signal[:u]
        s3=s1-estimated_f_t
        evm_sum=np.sum(np.abs(s3))
        cmplx_evm=(20*np.log10(evm_sum))/np.abs(u)
        magn_evm=((np.real(cmplx_evm))**2+(np.imag(cmplx_evm))**2)**0.5

        print("\nError Vector Magnitude (EVM) Model Estimation for CFO Est. Signal Done. \n")

        return magn_evm

    def threshold_est(self, estimated_f_t):

        ratio=estimated_f_t/self.f_t
        ll_ratio=(estimated_f_t-self.threshold_value)/self.f_t
        ul_ratio=(estimated_f_t+self.threshold_value)/self.f_t

        if ll_ratio<=ratio<=ul_ratio:
            valid_bit=1
        else:
            valid_bit=0

        return ratio*valid_bit

    #combined method calls

    def cfo_analysis(self, signal, ch, ch_b):

        print("\nAnalyzing CFO estimates (assuming no phase-wrapping)...\n")

        self.offsets.append(self.offset_to_hz(self.cfo_est_aom_nr(signal)))
        self.offsets.append(self.offset_to_hz(self.cfo_est_moa_nr(signal)))
        self.offsets.append(self.offset_to_hz(self.cfo_est_aom_r(signal)))
        self.offsets.append(self.offset_to_hz(self.cfo_est_moa_r(signal))) #offsets
        self.offsets.append(self.offset_to_hz(self.cfo_est_conventional(signal)))
        self.offsets.append(self.offset_to_hz(self.cfo_est_mle(signal)))
        self.offsets.append(self.offset_to_hz(self.cfo_est_blue(signal)))

        self.std_arr.append(np.abs(self.f_t-self.offsets[0]))
        self.mse_arr.append(self.capital_f(((np.abs(self.std_arr[0]))**2+0.5*(np.abs(self.std_arr[0]))**4)/(self.m*self.b)))
        self.crb_arr.append(self.crb_est(np.abs(self.std_arr[0])))

        self.std_arr.append(np.abs(self.f_t-self.offsets[1]))
        self.mse_arr.append(self.capital_f(((np.abs(self.std_arr[1]))**2+0.5*(np.abs(self.std_arr[1]))**4)/(self.m))/self.b)
        self.crb_arr.append(self.crb_est(np.abs(self.std_arr[1])))

        self.std_arr.append(np.abs(self.f_t-self.offsets[2]))
        self.mse_arr.append(self.capital_f((((np.abs(self.std_arr[2]))**2)/self.b+0.5*(np.abs(self.std_arr[2]))**4)/(self.m*self.b)))
        self.crb_arr.append(self.crb_est(np.abs(self.std_arr[2])))

        self.std_arr.append(np.abs(self.f_t-self.offsets[3]))
        self.mse_arr.append(self.capital_v(self.m, (np.abs(self.std_arr[3]))**2))
        self.crb_arr.append(self.crb_est(np.abs(self.std_arr[3])))

        self.std_arr.append(np.abs(self.f_t-self.offsets[4])) #STD
        self.mse_arr.append((self.std_arr[4])**2) #MSE
        self.crb_arr.append(self.crb_est(np.abs(self.std_arr[4]))) #CRLB 

        self.std_arr.append(np.abs(self.f_t-self.offsets[5])) #offset
        self.mse_arr.append((self.std_arr[5])**2) #MSE
        self.crb_arr.append(self.crb_est(np.abs(self.std_arr[5]))) #CRLB 

        self.std_arr.append(np.abs(self.f_t-self.offsets[6])) #offset
        self.mse_arr.append((self.std_arr[6])**2) #MSE
        self.crb_arr.append(self.crb_blue_est(signal,np.abs(self.std_arr[6]), ch_b)) #CRLB 

        self.cpe.append(self.cpe_est(signal, self.offsets[0],"nr"))
        self.cpe.append(self.cpe_est(signal, self.offsets[1],"nr"))
        self.cpe.append(self.cpe_est(signal, self.offsets[2],"r"))
        self.cpe.append(self.cpe_est(signal, self.offsets[3],"r"))
        self.cpe.append(self.cpe_est(signal, self.offsets[4],"nr"))
        self.cpe.append(self.cpe_est(signal, self.offsets[5],"nr"))
        self.cpe.append(self.cpe_est(signal, self.offsets[6],"nr"))

        self.evm.append(self.evm_est(signal, self.offsets[0], ch))
        self.evm.append(self.evm_est(signal, self.offsets[1], ch))
        self.evm.append(self.evm_est(signal, self.offsets[2], ch))
        self.evm.append(self.evm_est(signal, self.offsets[3], ch))
        self.evm.append(self.evm_est(signal, self.offsets[4], ch))
        self.evm.append(self.evm_est(signal, self.offsets[5], ch))
        self.evm.append(self.evm_est(signal, self.offsets[6], ch))

        self.threshold.append(self.threshold_est(self.offsets[0]))
        self.threshold.append(self.threshold_est(self.offsets[1]))
        self.threshold.append(self.threshold_est(self.offsets[2]))
        self.threshold.append(self.threshold_est(self.offsets[3]))
        self.threshold.append(self.threshold_est(self.offsets[4]))
        self.threshold.append(self.threshold_est(self.offsets[5]))
        self.threshold.append(self.threshold_est(self.offsets[6]))

        self.mse_arr=np.array(self.mse_arr)
        self.std_arr=np.array(self.std_arr)
        self.crb_arr=np.array(self.crb_arr)
        self.types_arr=np.array(self.types_arr)
        self.offsets=np.array(self.offsets) #normalized offset per generic smapling formula can be calculated but was ignored
        self.cpe=np.array(self.cpe) #CPE per generic formula for sampling period can be computed but is ignored. 
        self.evm=np.array(self.evm) #IT IS POSSIBLE TO ADD ABOVE IN A LOOP OF len(offsets) AND ADD CONDITIONALS (if i==1) TO DISCERN SLIGHT DIFFERENCES IN METHODICAL EVALUATIONS
        self.threshold=np.array(self.threshold)

        self.min_mse_val=self.mse_arr[np.where(np.abs(self.mse_arr)==np.min(np.abs(self.mse_arr)))[0][0]]
        self.min_mse_type=self.types_arr[np.where(np.abs(self.mse_arr)==np.min(np.abs(self.mse_arr)))[0][0]]

        self.min_std_val=self.std_arr[np.where(np.abs(self.std_arr)==np.min(np.abs(self.std_arr)))[0][0]]
        self.min_std_type=self.types_arr[np.where(np.abs(self.std_arr)==np.min(np.abs(self.std_arr)))[0][0]]

        self.min_crb_val=self.crb_arr[np.where(np.abs(self.crb_arr)==np.min(np.abs(self.crb_arr)))[0][0]]
        self.min_crb_type=self.types_arr[np.where(np.abs(self.crb_arr)==np.min(np.abs(self.crb_arr)))[0][0]]

        self.min_cpe_val=self.cpe[np.where(np.abs(self.cpe)==np.min(np.abs(self.cpe)))[0][0]]
        self.min_cpe_type=self.types_arr[np.where(np.abs(self.cpe)==np.min(np.abs(self.cpe)))[0][0]]

        self.min_evm_val=self.evm[np.where(np.abs(self.evm)==np.min(np.abs(self.evm)))[0][0]]
        self.min_evm_type=self.types_arr[np.where(np.abs(self.evm)==np.min(np.abs(self.evm)))[0][0]]

        self.min_thrshold_val=self.threshold[np.where(np.abs(self.threshold)==np.min(np.abs(self.threshold)))[0][0]]
        self.min_threshold_type=self.types_arr[np.where(np.abs(self.threshold)==np.min(np.abs(self.threshold)))[0][0]]

        self.min_threshold_offset=self.offsets[np.where(np.abs(self.threshold)==np.min(np.abs(self.threshold)))[0][0]]
        self.min_mse_offset=self.offsets[np.where(np.abs(self.mse_arr)==np.min(np.abs(self.mse_arr)))[0][0]]
        self.min_crb_offset=self.offsets[np.where(np.abs(self.crb_arr)==np.min(np.abs(self.crb_arr)))[0][0]]
        self.min_evm_offset=self.offsets[np.where(np.abs(self.evm)==np.min(np.abs(self.evm)))[0][0]]
        self.min_cpe_offset=self.offsets[np.where(np.abs(self.cpe)==np.min(np.abs(self.cpe)))[0][0]]
        self.min_std_offset=self.offsets[np.where(np.abs(self.std_arr)==np.min(np.abs(self.std_arr)))[0][0]]

        self.super_mse_arr.append(self.mse_arr)
        self.super_std_arr.append(self.std_arr)
        self.super_crb_arr.append(self.crb_arr)
        self.super_cpe_arr.append(self.cpe)
        self.super_evm_arr.append(self.evm)
        self.super_tso_arr.append(self.threshold)

        print("\nAnalysis Complete. Now this data can be used for  Autocorrelation + RSSI based Ping Detection.\n")

        pass

    def save_analysis(self): #save all analysis data to txt file

        filename="results/trial_{}_{}/datasave_trial_{}_{}.txt".format(self.stype, self.maxfilenum, self.stype, datetime.datetime.now())
        f=open(filename,"w")
        f.write("\n\nLogged Data From Experiment: \n\n")
        f.write("\n\nHyper-parameters considered in this experimental run (CPE correction or CPE non-correction): \n\n")
        f.write("\nSampling Frequency (Hz): {}\n".format(self.f_s))
        f.write("\nCarrier (Central) Frequency (Hz): {}\n".format(self.f_c))
        f.write("\nFrequency Offset to Guess (Hz): {}\n".format(self.f_t))
        f.write("\nPing Power (dB): {}\n".format(self.ping_power))
        f.write("\nNoise Power (dB): {}\n".format(self.noise_power))
        f.write("\nPilot Block Length (P): {}\n".format(self.p))
        f.write("\nPilot Block Length (M): {}\n".format(self.m))
        f.write("\nData Block Length (N): {}\n".format(self.n))
        f.write("\nNumber of Frames (B): {}\n".format(self.b))
        f.write("\nNumber of Frames (B', for Reuse): {}\n".format(self.b_dash))
        f.write("\nTransmission Block Length (Q): {}\n".format(self.q))
        f.write("\nDecimation Rate (ψ): {}\n".format(self.psi))
        f.write("\nBackoff Rate (ignored) (D): {}\n".format(self.d))
        f.write("\n\nSummary (Assumed: Bias=0, Transmitted Signal = Received Signal; for all methods): Signal Type: {}\n\n".format(self.stype))
        f.write("\nType of Analysis with Least MSE: {}, with a MSE (mean squared error) value of {} Hz, which is {} Hz for an actual offset of {} Hz\n".format(self.min_mse_type,self.min_mse_val, self.min_mse_offset, self.f_t))
        f.write("\nType of Analysis with Least STD DEV: {}, with a standard deviation value of {} Hz, which is {} Hz for an actual offset of {} Hz\n".format(self.min_std_type,self.min_std_val, self.min_std_offset, self.f_t))
        f.write("\nType of Analysis with Least Cramer Rao Lower Bound: {}, with a CRLB value of {} Hz, which is {} Hz for an actual offset of {} Hz\n".format(self.min_crb_type,self.min_crb_val, self.min_crb_offset, self.f_t))
        f.write("\nType of Analysis with Least CPE Offset: {}, with a corrected phase error value of {} rads, which is {} Hz for an actual offset of {} Hz\n".format(self.min_cpe_type,self.min_cpe_val, self.min_cpe_offset, self.f_t))
        f.write("\nType of Analysis with Least EVM value: {}, with a error vector magnitude value of {} dB, which is {} Hz for an actual offset of {} Hz\n".format(self.min_evm_type,self.min_evm_val, self.min_evm_offset, self.f_t))
        f.write("\nType of Analysis with Most Threshold value: {}, with a threshold value of {} dB, which is {} Hz for an actual offset of {} Hz\n".format(self.max_threshold_type,self.max_threshold_val, self.max_threshold_offset, self.f_t))
        f.write("\n\nMSE (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.mse_arr[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.mse_arr[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.mse_arr[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.mse_arr[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.mse_arr[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.mse_arr[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.mse_arr[6]))
        f.write("\n\nSTD DEV (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.std_arr[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.std_arr[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.std_arr[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.std_arr[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.std_arr[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.std_arr[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.std_arr[6]))
        f.write("\n\nCramer-Rao Lower Bound (CRLB) (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.crb_arr[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.crb_arr[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.crb_arr[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.crb_arr[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.crb_arr[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.crb_arr[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.crb_arr[6]))
        f.write("\n\nError Vector Magnitude (EVM) (dB): \n\n")
        f.write("\n{}: {} dB\n".format(self.types_arr[0], self.evm[0]))
        f.write("\n{}: {} dB\n".format(self.types_arr[1], self.evm[1]))
        f.write("\n{}: {} dB\n".format(self.types_arr[2], self.evm[2]))
        f.write("\n{}: {} dB\n".format(self.types_arr[3], self.evm[3]))
        f.write("\n{}: {} dB\n".format(self.types_arr[4], self.evm[4]))
        f.write("\n{}: {} dB\n".format(self.types_arr[5], self.evm[5]))
        f.write("\n{}: {} dB\n".format(self.types_arr[6], self.evm[6]))
        f.write("\n\nPhase Wrap Offset, Common Phase Error (CPE) (rads): \n\n")
        f.write("\n{}: {} rads\n".format(self.types_arr[0], self.cpe[0]))
        f.write("\n{}: {} rads\n".format(self.types_arr[1], self.cpe[1]))
        f.write("\n{}: {} rads\n".format(self.types_arr[2], self.cpe[2]))
        f.write("\n{}: {} rads\n".format(self.types_arr[3], self.cpe[3]))
        f.write("\n{}: {} rads\n".format(self.types_arr[4], self.cpe[4]))
        f.write("\n{}: {} rads\n".format(self.types_arr[5], self.cpe[5]))
        f.write("\n{}: {} rads\n".format(self.types_arr[6], self.cpe[6]))
        f.write("\n\nEstimated Offsets (CFO, or Carrier/Central Frequency Offsets) (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.offsets[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.offsets[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.offsets[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.offsets[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.offsets[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.offsets[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.offsets[6]))
        f.write("\n\nThresholds: (Ratio) \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.threshold[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.threshold[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.threshold[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.threshold[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.threshold[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.threshold[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.threshold[6]))
        f.write("\n\nLegend: \n\n")
        f.write("\nAOM-NR=Angle of Mean, Non-Reuse\n")
        f.write("\nMOA-NR=Mean of Angle, Non-Reuse\n")
        f.write("\nAOM-R=Angle of Mean, Reuse\n")
        f.write("\nMOA-R=Mean of Angle, Reuse\n")
        f.write("\nMLE=Maximum Likelihood Estimate \n")
        f.write("\nBLUE=Best Linear unbiased Estimator \n")
        f.write("\n\nUser Choice List of Experiment: \n\n")
        f.write("\nSignal Type for Experiments: {}\n".format(self.signal_type_print))
        f.write("\nB, Q Overwrite: {}\n".format(self.precompute_bq))
        f.write("\nHyperparameter Hardsetting: {}\n".format(self.hardset_hyper))
        f.write("\nType of Noise applied to Signal: {}\n".format(self.noise_applied))
        f.write("\nSignal Normalized with respect to Power: {}\n".format(self.sig_normalized))
        f.write("\nU Samples for EVM modeling: {}\n".format(self.utype_sample_evm))
        f.write("\nExperiment type: {}\n".format(self.experiment_type))
        f.close()

        print("\nData Saved into .txt file with appropriate subdirectory successfully. \n")

        pass

    #helper fns

    #---for analysis (MSE analysis):

    def offset_to_hz(self, x): #convert offsets computed to Hz value 

        return ((10**(-np.abs(x)))/20)*(10**5)

    def f_integrand(self, x, y): #main fn for F

        return ((2*y**2)*(np.exp((-(np.tan(y))**2)/(2*x))))/(((2*np.pi*x)**0.5)*((np.cos(y))**2))

    def capital_f(self, x):#function for mse 

        return (quad(self.f_integrand, 0, 0.5*np.pi, args=(x))[0])/((self.q**2)*(self.psi**2))

    def capital_v(self, x, y): #special function for mse for 1 case 

        return (self.capital_f(x)/self.b)+((2*(self.b-1)*self.capital_u(x,y))/((self.b**2)*(self.q**2)*(self.psi**2)))

    def u_integrand(self, u,v,x,y): #integrable for U

        return np.arctan(u)*np.arctan(v)*(x/(2*np.pi*(y+(0.5*y**2))*(1-(1/(2+y)**2))**0.5))*np.exp((-((x*(u**2+v**2+((2*u*v)/(2+y))))/(y+(0.5*y**2))))/(2*(1-(1/(2+y)**2))))

    def integral1(self, u,x,y): #helper for dbl integral 

        return quad(self.u_integrand,-np.inf,np.inf,args=(u,x,y))[0]

    def capital_u(self, x, y): #helper in v, as a 2D fn

        return quad(lambda u: self.integral1(u,x,y), -np.inf, np.inf)[0] #integrate v first, then u

    #---for hyperparam setting: 

    def hardset_hp(self, choice, choice1, choice2, choice3, choice4): #set hyperparams to experimental values 

        n_arr=np.array([256,512,1024,2048])
        m_arr=np.array([16,32,64,128])
        b_arr=np.array([13,27,55,127])
        
        if choice==0:
            self.m=32
            self.n=128
            self.q=self.m+self.n
            self.b=88
            
        else:
            self.n=n_arr[choice1]
            self.m=m_arr[choice2]
            self.b=b_arr[choice3]
            self.q=self.m+self.n

        if choice4==0:
            self.b_dash=self.b+1
        else:
            self.b_dash=0.5*(self.b+1)
        self.p=self.m
        self.psi=4 #ignore d=4

        print("\nHyperparameters Set. \n")

        pass

    def b_effective(self, w): #b_eff resets

        self.b= (self.n*self.b)/(w*self.n+(w-1)*self.m)
        self.b=int(self.b)
        self.b_dash=self.b+1

        print("\nB Effective Set. \n")

        pass

    def q_effective(self, w): #q_eff resets 

        self.q= w*self.n+(w-1)*self.m

        print("\nQ Effective Set. \n")

        pass

    #optional calls:

    def add_noise(self, signal, noise_type): #adds AWGn, Complex Zero Mean Noise, or None

        signal_wn=[]
        if noise_type=="czmrgn":
            signal_wn=signal+np.random.normal(0, 0.5*np.std(signal)**2, len(signal))+1j*np.random.normal(0, 0.5*np.std(signal)**2, len(signal))
        elif noise_type=="awgn":
            snr_db=10.0**(self.noise_power/10.0)
            gamma = 10**(snr_db/10) #SNR to linear scale
            pp=np.sum(np.abs(signal)**2)/len(signal)
            n0=pp/gamma
            signal_wn=signal+np.sqrt(n0/2)*np.random.standard_normal(len(signal))+1j*np.sqrt(n0/2)*(np.random.standard_normal(len(signal))+1j*np.random.standard_normal(len(signal)))

        print("\nNoise added to signal. \n")

        return np.array(signal_wn)

    # other functions for non-single experiments:

    def pre_set_hypers(self): #"best" fit hyper params for the multi-experiments 
        self.n=1024
        self.m=64
        self.b=55
        self.q=self.m+self.n
        self.b_dash=self.b+1
        self.p=self.m
        self.psi=4 #ignore d=4
        w=6 
        self.q= w*self.n+(w-1)*self.m
        self.b= (self.n*self.b)/(w*self.n+(w-1)*self.m)
        self.b=int(self.b)
        self.b_dash=self.b+1
        print("\nPre-Set Hyper Parameters Set. \n")

    def cfo_est_blue_sample(self, signal, sample_i): #BLUE method for changing sample 

        total_n=500 #len(signal)
        u=int(sample_i)
        r=int(total_n/u)
        k=int(u/2)
        fsum=0
        last_phiu=-1
        phiu=-1
        for i in range(k):
            mu=np.arange(i*r, total_n)
            mu2=mu-i*r
            a4=np.where(mu<len(signal),mu,mu-len(signal))
            a5=np.where(mu2<len(signal),mu2,mu2-len(signal))
            a4=np.where(a4<len(signal), a4, len(signal)-1)
            a5=np.where(a5<len(signal), a5, len(signal)-1)
            a4=np.where(a4<len(signal), a4, len(signal)-1)
            a5=np.where(a5<len(signal), a5, len(signal)-1)
            
            s1=signal[a4] 
            s2=signal[a5]
            phi_bu=np.sum(self.autocorrelate_phi(s1, s2))/(total_n-i*r)
            if u>=1:
                phiu=(np.arctan(np.imag(phi_bu)/np.real(phi_bu))-np.arctan(np.imag(last_phiu)/np.real(last_phiu)))%(2*np.pi)
            else:
                phiu=phi_bu
            last_phiu=phiu
            wu=((3*(u-i)*(u-i+1))-(k*(u-k)))/(k*(4*k*k-6*u*k+3*u*u-1))
            fsum+=(wu*phiu)

        print("\nBLUE Method CFO Est. with sample of {} Done. \n".format(sample_i))

        return 1/((fsum*u*self.f_s)/(2*np.pi))

    def cfo_est_aom_r_sample(self,signal, sample_i): #angle of mean with reuse for changing sample 

        sum_aom_r=0
        k=np.arange(1,self.p+1)
        for i in range(1, sample_i+1):
            a1=self.q*i-self.q+k
            a2=self.q*i+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a4=np.where(a4<len(signal), a4, len(signal)-1)
            a5=np.where(a5<len(signal), a5, len(signal)-1)
            a4=np.where(a4<len(signal), a4, len(signal)-1)
            a5=np.where(a5<len(signal), a5, len(signal)-1)
            s1=signal[a4] 
            s2=signal[a5]
            sum_aom_r+=(np.sum(self.autocorrelate_phi(s1, s2)))
        ang_sig=sum_aom_r/(self.m*self.b)

        angle=np.arctan(np.imag(ang_sig)/np.real(ang_sig))

        print("\nAngle of Mean - Reuse method for CFO Est. with sample of {} Done. \n".format(sample_i))

        return angle/(self.psi*self.q)

    def cfo_est_aom_nr_sample(self, signal, sample_i): #angle of mean with no reuse for changing sample 
    
        sum_aom_nr=0
        k=np.arange(1,self.p+1)
        for i in range(1, sample_i+1):
            a1=2*self.q*i-2*self.q+k
            a2=2*self.q*i-self.q+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal)) 
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            a6=np.where(a6<len(signal), a6, len(signal)-1)
            a7=np.where(a7<len(signal), a7, len(signal)-1)
            s1=signal[a6] 
            s2=signal[a7]
            sum_aom_nr+=(np.sum(self.autocorrelate_phi(s1,s2)))
        ang_sig=sum_aom_nr/(self.m*self.b)

        angle=np.arctan(np.imag(ang_sig)/np.real(ang_sig))

        print("\nAngle of Mean - NonReuse method for CFO Est. with sample of {} Done. \n".format(sample_i))

        return angle/(self.psi*self.q)

    def cfo_est_moa_r_sample(self, signal, sample_i): #mean of angle with reuse for changing sample 
    
        angle_sum=0
        k=np.arange(1,self.p+1)
        for i in range(1, sample_i+1):
            a1=self.q*i-self.q+k
            a2=self.q*i+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            a6=np.where(a6<len(signal), a6, a6-len(signal)-1)
            a7=np.where(a7<len(signal), a7, a7-len(signal)-1)
            s1=signal[a6] 
            s2=signal[a7]
            sum_moa_r=(np.sum(self.autocorrelate_phi(s1,s2)))/self.m
            angle=np.arctan(np.imag(sum_moa_r)/np.real(sum_moa_r))
            angle_sum+=angle 

        print("\nMean of Angle - Reuse method for CFO Est. with sample of {} Done. \n".format(sample_i))

        return angle_sum/(self.psi*self.q*self.b)

    def cfo_est_moa_nr_sample(self, signal, sample_i): #mean of angle without reuse for changing sample 
    
        angle_sum=0
        k=np.arange(1,self.p+1)
        for i in range(1, sample_i+1):
            a1=2*self.q*i-2*self.q+k
            a2=2*self.q*i-self.q+k
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            a6=np.where(a6<len(signal), a6, len(signal)-1)
            a7=np.where(a7<len(signal), a7, len(signal)-1)
            s1=signal[a6] 
            s2=signal[a7]
            sum_moa_nr=(np.sum(self.autocorrelate_phi(s1,s2)))/self.m
            angle=np.arctan(np.imag(sum_moa_nr)/np.real(sum_moa_nr))
            angle_sum+=angle 

        print("\nMean of Angle - NonReuse method for CFO Est. with sample of {} Done. \n".format(sample_i))

        return angle_sum/(self.psi*self.q*self.b)

    def cfo_est_conventional_sample(self, signal, sample_i): #typical method for cfo for changing sample 
    
        l_dash=int((len(signal)-1)/self.d)
        sum_phi=0
        i=np.arange(0,sample_i-1)
        a1=self.q*i+l_dash
        a2=self.q*i+l_dash+self.q
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal))
        a4=np.where(a4<len(signal), a4, len(signal)-1)
        a5=np.where(a5<len(signal), a5, len(signal)-1)
        a4=np.where(a4<len(signal), a4, len(signal)-1)
        a5=np.where(a5<len(signal), a5, len(signal)-1)
        s1=signal[a4] 
        s2=signal[a5]
        sum_phi=np.sum(self.autocorrelate_phi(s1,s2))

        print("\nConventional method for CFO Est. with sample of {} Done. \n".format(sample_i))

        return (np.arctan((np.imag(sum_phi))/(self.m*(np.real(sum_phi)))))/(self.q*self.psi)

    def cfo_est_mle_sample(self, signal, sample_i): #MLE method for chanigns sample 

        l_dash=int((len(signal)-1)/self.d)
        w_mle=0
        for mu in range(1, sample_i+1):
            sum_mle=0
            p=np.arange(mu, self.b+1)
            a1=l_dash+self.q*(p-mu)
            a2=l_dash+self.q*(p-mu)+mu*self.q 
            a4=np.where(a1<len(signal),a1,a1-len(signal))
            a5=np.where(a2<len(signal),a2,a2-len(signal))
            a6=np.where(a4<len(signal),a4,len(signal)-1)
            a7=np.where(a5<len(signal),a5,len(signal)-1)
            a6=np.where(a4<len(signal),a6,len(signal)-1)
            a7=np.where(a5<len(signal),a7,len(signal)-1)
            s1=signal[a6] 
            s2=signal[a7]
            sum_mle=np.sum(self.autocorrelate_phi(s1,s2))
            angle=np.arctan(np.imag(sum_mle)/np.real(sum_mle))
            w_mle+=(angle/mu)

        print("\nMLE method for CFO Est. with sample of {} Done. \n".format(sample_i))

        return w_mle/(self.q*self.psi) 

    def test_by_sample(self, signal, increment_value, fac_i): #changes sample to an instance for testing

        c=0 #counter

        for i in range(5,int(len(signal)/fac_i), increment_value):

            print("\nAnalyzing CFO estimates (assuming no phase-wrapping), for {} samples...\n".format(i))

            self.offsets.append(self.offset_to_hz(self.cfo_est_aom_nr_sample(signal,i)))
            self.offsets.append(self.offset_to_hz(self.cfo_est_moa_nr_sample(signal,i)))
            self.offsets.append(self.offset_to_hz(self.cfo_est_aom_r_sample(signal,i)))
            self.offsets.append(self.offset_to_hz(self.cfo_est_moa_r_sample(signal,i))) #offsets
            self.offsets.append(self.offset_to_hz(self.cfo_est_conventional_sample(signal,i)))
            self.offsets.append(self.offset_to_hz(self.cfo_est_mle_sample(signal,i)))
            self.offsets.append(self.offset_to_hz(self.cfo_est_blue_sample(signal,i)))

            self.std_arr.append(np.abs(self.f_t-self.offsets[0]))
            self.mse_arr.append(self.capital_f(((np.abs(self.std_arr[0]))**2+0.5*(np.abs(self.std_arr[0]))**4)/(self.m*self.b)))
            self.crb_arr.append(self.crb_est(np.abs(self.std_arr[0])))

            self.std_arr.append(np.abs(self.f_t-self.offsets[1]))
            self.mse_arr.append(self.capital_f(((np.abs(self.std_arr[1]))**2+0.5*(np.abs(self.std_arr[1]))**4)/(self.m))/self.b)
            self.crb_arr.append(self.crb_est(np.abs(self.std_arr[1])))

            self.std_arr.append(np.abs(self.f_t-self.offsets[2]))
            self.mse_arr.append(self.capital_f((((np.abs(self.std_arr[2]))**2)/self.b+0.5*(np.abs(self.std_arr[2]))**4)/(self.m*self.b)))
            self.crb_arr.append(self.crb_est(np.abs(self.std_arr[2])))

            self.std_arr.append(np.abs(self.f_t-self.offsets[3]))
            self.mse_arr.append(self.capital_v(self.m, (np.abs(self.std_arr[3]))**2))
            self.crb_arr.append(self.crb_est(np.abs(self.std_arr[3])))

            self.std_arr.append(np.abs(self.f_t-self.offsets[4])) #STD
            self.mse_arr.append((self.std_arr[4])**2) #MSE
            self.crb_arr.append(self.crb_est(np.abs(self.std_arr[4]))) #CRLB 

            self.std_arr.append(np.abs(self.f_t-self.offsets[5])) #offset
            self.mse_arr.append((self.std_arr[5])**2) #MSE
            self.crb_arr.append(self.crb_est(np.abs(self.std_arr[5]))) #CRLB 

            self.std_arr.append(np.abs(self.f_t-self.offsets[6])) #offset
            self.mse_arr.append((self.std_arr[6])**2) #MSE
            self.crb_arr.append(self.crb_blue_est(signal,np.abs(self.std_arr[6]), 1)) #CRLB 

            self.cpe.append(self.cpe_est(signal, self.offsets[0],"nr"))
            self.cpe.append(self.cpe_est(signal, self.offsets[1],"nr"))
            self.cpe.append(self.cpe_est(signal, self.offsets[2],"r"))
            self.cpe.append(self.cpe_est(signal, self.offsets[3],"r"))
            self.cpe.append(self.cpe_est(signal, self.offsets[4],"nr"))
            self.cpe.append(self.cpe_est(signal, self.offsets[5],"nr"))
            self.cpe.append(self.cpe_est(signal, self.offsets[6],"nr"))

            self.evm.append(self.evm_est(signal, self.offsets[0], 0))
            self.evm.append(self.evm_est(signal, self.offsets[1], 0))
            self.evm.append(self.evm_est(signal, self.offsets[2], 0))
            self.evm.append(self.evm_est(signal, self.offsets[3], 0))
            self.evm.append(self.evm_est(signal, self.offsets[4], 0))
            self.evm.append(self.evm_est(signal, self.offsets[5], 0))
            self.evm.append(self.evm_est(signal, self.offsets[6], 0))

            self.threshold.append(self.threshold_est(self.offsets[0]))
            self.threshold.append(self.threshold_est(self.offsets[1]))
            self.threshold.append(self.threshold_est(self.offsets[2]))
            self.threshold.append(self.threshold_est(self.offsets[3]))
            self.threshold.append(self.threshold_est(self.offsets[4]))
            self.threshold.append(self.threshold_est(self.offsets[5]))
            self.threshold.append(self.threshold_est(self.offsets[6]))

            self.offsets=np.nan_to_num(self.offsets, nan=np.inf)
            self.mse_arr=np.nan_to_num(self.mse_arr, nan=np.inf)
            self.crb_arr=np.nan_to_num(self.crb_arr, nan=np.inf)
            self.std_arr=np.nan_to_num(self.std_arr, nan=np.inf)
            self.evm=np.nan_to_num(self.evm, nan=np.inf)
            self.cpe=np.nan_to_num(self.cpe, nan=np.inf)
            self.threshold=np.nan_to_num(self.threshold, nan=np.inf)

            self.mse_arr=np.array(self.mse_arr)
            self.std_arr=np.array(self.std_arr)
            self.crb_arr=np.array(self.crb_arr)
            self.types_arr=np.array(self.types_arr)
            self.offsets=np.array(self.offsets) #normalized offset per generic smapling formula can be calculated but was ignored
            self.cpe=np.array(self.cpe) #CPE per generic formula for sampling period can be computed but is ignored. 
            self.evm=np.array(self.evm) #IT IS POSSIBLE TO ADD ABOVE IN A LOOP OF len(offsets) AND ADD CONDITIONALS (if i==1) TO DISCERN SLIGHT DIFFERENCES IN METHODICAL EVALUATIONS
            self.threshold=np.array(self.threshold)

            self.sample_mse_arr.append(self.mse_arr)
            self.sample_std_arr.append(self.std_arr)
            self.sample_crb_arr.append(self.crb_arr)
            self.sample_offsets.append(self.offsets)
            self.sample_cpe.append(self.cpe)
            self.sample_evm.append(self.evm)
            self.sample_threshold.append(self.threshold)
            
            a=self.sample_mse_arr[c]
            a=np.nan_to_num(a, nan=np.inf)
            a0=np.min(a)
            a=np.where(a==np.inf, a, a0*1.01)

            t_a=-1
            for a_i in range(len(a)):
                if a[a_i]==a0:
                    t_a=a_i
            types_a=self.types_arr[t_a]
            off_a=self.offsets[t_a]

            self.min_mse_val=a0
            self.min_mse_type=types_a
            self.min_mse_offset=off_a

            a=self.sample_std_arr[c]
            a=np.nan_to_num(a, nan=np.inf)
            a0=np.min(a)
            a=np.where(a==np.inf, a, a0*1.01)

            t_a=-1
            for a_i in range(len(a)):
                if a[a_i]==a0:
                    t_a=a_i
            types_a=self.types_arr[t_a]
            off_a=self.offsets[t_a]

            self.min_std_val=a0
            self.min_std_type=types_a
            self.min_std_offset=off_a

            a=self.sample_crb_arr[c]
            a=np.nan_to_num(a, nan=np.inf)
            a0=np.min(a)
            a=np.where(a==np.inf, a, a0*1.01)

            t_a=-1
            for a_i in range(len(a)):
                if a[a_i]==a0:
                    t_a=a_i
            types_a=self.types_arr[t_a]
            off_a=self.offsets[t_a]

            self.min_crb_val=a0
            self.min_crb_type=types_a
            self.min_crb_offset=off_a

            a=self.sample_cpe[c]
            a=np.nan_to_num(a, nan=np.inf)
            a0=np.min(a)
            a=np.where(a==np.inf, a, a0*1.01)

            t_a=-1
            for a_i in range(len(a)):
                if a[a_i]==a0:
                    t_a=a_i
            types_a=self.types_arr[t_a]
            off_a=self.offsets[t_a]

            self.min_cpe_val=a0
            self.min_cpe_type=types_a
            self.min_cpe_offset=off_a

            a=self.sample_evm[c]
            a=np.nan_to_num(a, nan=np.inf)
            a0=np.min(a)
            a=np.where(a==np.inf, a, a0*1.01)

            t_a=-1
            for a_i in range(len(a)):
                if a[a_i]==a0:
                    t_a=a_i
            types_a=self.types_arr[t_a]
            off_a=self.offsets[t_a]

            self.min_evm_val=a0
            self.min_evm_type=types_a
            self.min_evm_offset=off_a

            a=self.sample_threshold[c]
            a=np.nan_to_num(a, nan=np.inf)
            a0=np.min(a)
            a=np.where(a==np.inf, a, a0*1.01)

            t_a=-1
            for a_i in range(len(a)):
                if a[a_i]==a0:
                    t_a=a_i
            types_a=self.types_arr[t_a]
            off_a=self.offsets[t_a]

            self.max_threshold_val=a0
            self.max_threshold_type=types_a
            self.max_threshold_offset=off_a

            self.save_analysis_sample(i)

            self.mse_arr=[] #reset
            self.std_arr=[]
            self.crb_arr=[]
            self.offsets=[]
            self.cpe=[]
            self.evm=[]
            self.threshold=[]

            self.sample_vals.append(i)

            print("\nAnalysis Complete. Now this data can be used for  Autocorrelation + RSSI based Ping Detection. This is for {} samples.\n".format(i))

            c+=1

        self.sample_mse_arr=np.array(self.sample_mse_arr)
        self.sample_std_arr=np.array(self.sample_std_arr)
        self.sample_crb_arr=np.array(self.sample_crb_arr)
        self.sample_offsets=np.array(self.sample_offsets) 
        self.sample_cpe=np.array(self.sample_cpe) 
        self.sample_evm=np.array(self.sample_evm)
        self.sample_threshold=np.array(self.sample_threshold)
        self.sample_vals=np.array(self.sample_vals)

        print("\nOverall analysis for all sample changes complete. \n")

        pass

    def plot_crb_samples(self, i): #crb for each method vs sample len

        for j in range(len(self.sample_crb_arr)):
            sca=self.sample_crb_arr[j]
            self.crb_i.append(sca[i])

        self.crb_i=np.array(self.crb_i)
        self.crb_ii=self.crb_i

        plt.plot(self.sample_vals, self.crb_i, color="red")
        plt.title("Sample Length (Unit) vs. Cramer Rao Lower Bound (CRB) (Hz) for {} Hz offset".format(self.f_t))
        plt.xlabel("Sample Length (Unit)")
        plt.ylabel("Cramer Rao Lower Bound (CRB) (Hz)")
        plt.savefig("results/trial_{}_{}/crb_samples_plot_{}_{}.png".format(self.stype, self.maxfilenum, self.types_arr[i], datetime.datetime.now()))

        self.crb_i=[]

        print("\nCRB vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_mse_samples(self, i): #mse for each method vs sample len

        for j in range(len(self.sample_mse_arr)):
            mca=self.sample_mse_arr[j]
            self.mse_i.append(mca[i])

        self.mse_i=np.array(self.mse_i)
        self.mse_ii=self.mse_i

        plt.plot(self.sample_vals, self.mse_i, color="blue")
        plt.title("Sample Length (Unit) vs. Mean Squared Error (MSE) (Hz) for {} Hz offset".format(self.f_t))
        plt.xlabel("Sample Length (Unit)")
        plt.ylabel("Mean Squared Error (MSE) (Hz)")
        plt.savefig("results/trial_{}_{}/mse_samples_plot_{}_{}.png".format(self.stype, self.maxfilenum,self.types_arr[i], datetime.datetime.now()))

        self.mse_i=[]

        print("\nMSE vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_evm_samples(self, i): #evm for each method vs sample len

        for j in range(len(self.sample_evm)):
            eca=self.sample_evm[j]
            self.evm_i.append(eca[i])

        self.evm_i=np.array(self.evm_i)
        self.evm_ii=self.evm_i

        plt.plot(self.sample_vals, self.evm_i, color="green")
        plt.title("Sample Length (Unit) vs. Error Vector Magnitude (EVM) (dB) for {} Hz offset".format(self.f_t))
        plt.xlabel("Sample Length (Unit)")
        plt.ylabel("Error Vector Magnitude (EVM) (dB)")
        plt.savefig("results/trial_{}_{}/evm_samples_plot_{}_{}.png".format(self.stype, self.maxfilenum, self.types_arr[i], datetime.datetime.now()))

        self.evm_i=[]

        print("\nEVM vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_cpe_samples(self, i): #cpe for each method vs sample len

        for j in range(len(self.sample_cpe)):
            cca=self.sample_cpe[j]
            self.cpe_i.append(cca[i])

        self.cpe_i=np.array(self.cpe_i)
        self.cpe_ii=self.cpe_i

        plt.plot(self.sample_vals, self.cpe_i, color="orange")
        plt.title("Sample Length (Unit) vs. Common/Corrected Phase Error (CPE) (rads) for {} Hz offset".format(self.f_t))
        plt.xlabel("Sample Length (Unit)")
        plt.ylabel("Corrected Phase Error (CPE) (rads)")
        plt.savefig("results/trial_{}_{}/cpe_samples_plot_{}_{}.png".format(self.stype, self.maxfilenum, self.types_arr[i], datetime.datetime.now()))

        self.cpe_i=[]

        print("\nCPE vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_std_samples(self, i): #std for each method vs sample len

        for j in range(len(self.sample_std_arr)):
            tca=self.sample_std_arr[j]
            self.std_i.append(tca[i])

        self.std_i=np.array(self.std_i)
        self.std_ii=self.std_i

        plt.plot(self.sample_vals, self.std_i, color="cyan")
        plt.title("Sample Length (Unit) vs. Standard Deviation (STD) (Hz) for {} Hz offset".format(self.f_t))
        plt.xlabel("Sample Length (Unit)")
        plt.ylabel("Standard Deviation (STD) (Hz)")
        plt.savefig("results/trial_{}_{}/std_samples_plot_{}_{}.png".format(self.stype, self.maxfilenum, self.types_arr[i], datetime.datetime.now()))

        self.std_i=[]

        print("\nSTD vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_threshold_samples(self, i):#threshold for each method vs sample len

        for j in range(len(self.sample_threshold)):
            hca=self.sample_threshold[j]
            self.tso_i.append(hca[i])

        self.tso_i=np.array(self.tso_i)
        self.tso_ii=self.tso_i

        plt.plot(self.sample_vals, self.tso_i, color="pink")
        plt.title("Sample Length (Unit) vs. Threshold (TSO) (RATIO) for {} Hz offset".format(self.f_t))
        plt.xlabel("Sample Length (Unit)")
        plt.ylabel("Threshold (TSO) (Ratio)")
        plt.savefig("results/trial_{}_{}/tso_samples_plot_{}_{}.png".format(self.stype, self.maxfilenum, self.types_arr[i], datetime.datetime.now()))
        
        self.tso_i=[]

        print("\nTSO vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_crb_offset(self, i): #crb for each method vs offset

        for j in range(len(self.super_crb_arr)):
            hca=self.super_crb_arr[j]
            self.crb_j.append(hca[i])

        self.crb_j=np.array(self.crb_j)
        self.crb_jj=self.tso_j

        plt.plot(self.offset_vals, self.crb_j)
        plt.title("Offset Value (Hz) vs. Cramer Rao Lower Bound (CRB) (Hz) for {} sample".format(self.f_s))
        plt.xlabel("Offset Value (Hz)")
        plt.ylabel("Cramer Rao Lower Bound (CRB) (Hz)")
        plt.savefig("results/trial_{}_{}/crb_offsets_plot_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))
        
        self.crb_j=[]

        print("\nCRB vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_mse_offset(self, i): #mse for each method vs offset

        for j in range(len(self.super_mse_arr)):
            hca=self.super_mse_arr[j]
            self.mse_j.append(hca[i])

        self.mse_j=np.array(self.mse_j)
        self.mse_jj=self.mse_j

        plt.plot(self.offset_vals, self.mse_j)
        plt.title("Offset Value (Hz) vs. Mean Squared Error (MSE) (Hz) for {} sample".format(self.f_s))
        plt.xlabel("Offset Value (Hz)")
        plt.ylabel("Mean Squared Error (MSE) (Hz)")
        plt.savefig("results/trial_{}_{}/mse_offsets_plot_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))
        
        self.mse_j=[]

        print("\nMSE vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_evm_offset(self, i): #evm for each method vs offset

        for j in range(len(self.super_evm_arr)):
            hca=self.super_evm_arr[j]
            self.evm_j.append(hca[i])

        self.evm_j=np.array(self.evm_j)
        self.evm_jj=self.evm_j

        plt.plot(self.offset_vals, self.evm_j)
        plt.title("Offset Value (Hz) vs. Error Vector Magnitude (EVM) (dB) for {} sample".format(self.f_s))
        plt.xlabel("Offset Value (Hz)")
        plt.ylabel("Error Vector Magnitude (EVM) (dB)")
        plt.savefig("results/trial_{}_{}/evm_offsets_plot_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))
        
        self.evm_j=[]

        print("\nEVM vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_cpe_offset(self, i): #cpe for each method vs offset

        for j in range(len(self.super_cpe_arr)):
            hca=self.super_cpe_arr[j]
            self.cpe_j.append(hca[i])

        self.cpe_j=np.array(self.cpe_j)
        self.cpe_jj=self.cpe_j

        plt.plot(self.offset_vals, self.cpe_j)
        plt.title("Offset Value (Hz) vs. Common Phase Error (CPE) (rads) for {} sample".format(self.f_s))
        plt.xlabel("Offset Value (Hz)")
        plt.ylabel("Common Phase Error (CPE) (rads)")
        plt.savefig("results/trial_{}_{}/cpe_offsets_plot_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))
        
        self.cpe_j=[]

        print("\nCPE vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_std_offset(self, i): #std for each method vs offset

        for j in range(len(self.super_std_arr)):
            hca=self.super_std_arr[j]
            self.std_j.append(hca[i])

        self.std_j=np.array(self.std_j)
        self.std_jj=self.std_j

        plt.plot(self.offset_vals, self.std_j)
        plt.title("Offset Value (Hz) vs. Standard Deviation (STD) (Hz) for {} sample".format(self.f_s))
        plt.xlabel("Offset Value (Hz)")
        plt.ylabel("Standard Deviation (STD) (Hz)")
        plt.savefig("results/trial_{}_{}/std_offsets_plot_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))
        
        self.std_j=[]

        print("\nSTD vs. Samples plotted, data saved in directory. \n") 

        pass

    def plot_threshold_offset(self, i):#threshold for each method vs offset

        for j in range(len(self.super_tso_arr)):
            hca=self.super_tso_arr[j]
            self.tso_j.append(hca[i])

        self.tso_j=np.array(self.tso_j)
        self.tso_jj=self.tso_j

        plt.plot(self.offset_vals, self.tso_j)
        plt.title("Offset Value (Hz) vs. Threshold (TSO) (Ratio) for {} sample".format(self.f_s))
        plt.xlabel("Offset Value (Hz)")
        plt.ylabel("Threshold (TSO) (Ratio)")
        plt.savefig("results/trial_{}_{}/tso_offsets_plot_{}.png".format(self.stype, self.maxfilenum, datetime.datetime.now()))
        
        self.std_j=[]

        print("\nSTD vs. Samples plotted, data saved in directory. \n") 

        pass

    def n_test_signals_sum(self, offset_arr, method): # n transmitters of test signals summed

        if method==0:

            _, sv=self.generate_test_signal()
            a=np.repeat(0+0j,len(sv))
            for i in range(len(offset_arr)):
                self.f_t=offset_arr[i]
                _, sv=self.generate_test_signal()
                self.f_t=offset_arr[i]
                a+=sv

            return a

        elif method==1:

            m_arr=[]

            for i in range(len(offset_arr)):
                self.f_t=offset_arr[i]
                _, sv=self.generate_test_signal()
                m_arr.append(sv)

            m_arr=np.array(m_arr)
            m_arr=m_arr.T

            return m_arr.dot(self.mixing_coeffs).T

    def custom_nmf(self, X, tol=1e-6, max_iter=5000): #custom emthod for NMF
    
        n_components=self.n_trans

        W = np.random.rand(X.shape[0], n_components)
        H = np.random.rand(n_components, X.shape[1])

        oldlim = 1e9

        eps = 1e-7

        for i in range(max_iter):
            H = H * ((W.T.dot(X) + eps) / (W.T.dot(W).dot(H) + eps))
            W = W * ((X.dot(H.T) + eps) / (W.dot(H.dot(H.T)) + eps))

            lim = np.linalg.norm(X-W.dot(H), 'fro')

            if abs(oldlim - lim) < tol:
                break

            oldlim = lim

        return W

    def custom_ica(self, X, step_size=1, tol=1e-8, max_iter=10000): #custom method for ICA

        n_sources=self.n_trans
        m, _ = X.shape

        W = np.random.rand(n_sources, m)
        
        for c in range(n_sources):
            w = W[c, :].copy().reshape(m, 1)
            w = w / np.sqrt((w ** 2).sum())

            for i in range(max_iter):
                v = np.dot(w.T, X)

                gv = np.tanh(v*step_size).T

                gdv = (1 - np.square(np.tanh(v))) * step_size

                wNew = (X * gv.T).mean(axis=1) - gdv.mean() * w.squeeze()

                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                lim = np.abs(np.abs((wNew * w).sum()) - 1)
                w = wNew
                if lim < tol:
                    break

            W[c, :] = w.T
        return W

    def ica(self, full_sig, method_opt): #ica separation for n test signals by ica method 

        if method_opt==0: #sklearn method 
            dec_sig_model=decomposition.FastICA(n_components=self.n_trans, algorithm='parallel', whiten='warn', fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None)
            transformed_data=dec_sig_model.fit_transform(full_sig)

            sep_sigs=[]
            for i in range(len(transformed_data)):
                sep_sigs.append(transformed_data[i])

            return np.array(sep_sigs)

        elif method_opt==1: #custom method 
            W=self.custom_ica(full_sig)

            sep_sigs=[]
            for i in range(len(W)):
                sep_sigs.append(W[i])

            return np.array(sep_sigs)

        elif method_opt==2: #another custom method 

            Xc= self.center(full_sig)
            Xw, _ = self.whiten(Xc)
            W = self.fastIca(Xw,  alpha=1)
            unMixed = Xw.T.dot(W.T)
            unMixed = (unMixed.T - np.mean(Xc)).T

            sep_sigs=[]
            for i in range(len(unMixed)):
                sep_sigs.append(unMixed[i])

            return np.array(sep_sigs)

        print('\nICA method of decomposition performed for the multi-threaded signal. \n')

        pass
    
    def kurtosis(self, x): #kurtosis of sample 

        n = np.shape(x)[0]
        mean = np.sum((x**1)/n) # Calculate the mean
        var = np.sum((x-mean)**2)/n # Calculate the variance
        skew = np.sum((x-mean)**3)/n # Calculate the skewness
        kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
        kurt = kurt/(var**2)-3

        return kurt, skew, var, mean

    def covariance(self, x): #cov for x custom 

        mean = np.mean(x, axis=1, keepdims=True)
        n = np.shape(x)[1] - 1
        m = x - mean

        return (m.dot(m.T))/n

    def whiten(self, X):

        coVarM = self.covariance(X) 

        U, S, V = np.linalg.svd(coVarM)
        
        d = np.diag(1.0 / np.sqrt(S)) 
        
        whiteM = np.dot(U, np.dot(d, U.T))
        
        Xw = np.dot(whiteM, X) 
        
        return Xw, whiteM

    def center(self, x): #center signal 
        return x - np.mean(x, axis=1, keepdims=True)

    def fastIca(self, signals,  alpha = 1, thresh=1e-8, iterations=5000): #custom fast ica method 
        m, n = signals.shape
        W = np.random.rand(m, m)
        for c in range(m):
            w = W[c, :].copy().reshape(m, 1)
            w = w/ np.sqrt((w ** 2).sum())
            i = 0
            lim = 100
            while ((lim > thresh) & (i < iterations)):
                ws = np.dot(w.T, signals)
                wg = np.tanh(ws * alpha).T
                wg_ = (1 - np.square(np.tanh(ws))) * alpha
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                w = wNew
                i += 1

            W[c, :] = w.T
        return W

    def nmf(self, full_sig, method_opt): #ica separation for n test signals by nmf method 

        if method_opt==0: #sklearn method 
            dec_sig_model=decomposition.NMF(n_components=self.n_trans, init="nndsvda", solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha="deprecated")
            W=dec_sig_model.fit_transform(full_sig)
            
            sep_sigs=[]
            for i in range(len(W)):
                sep_sigs.append(W[i])

            return np.array(sep_sigs)

        elif method_opt==1: #custom method 
            W=self.custom_nmf(full_sig)

            sep_sigs=[]
            for i in range(len(W)):
                sep_sigs.append(W[i])

            return np.array(sep_sigs)

        print('\nNMF method of decomposition performed for the multi-threaded signal. \n')

        pass

    def analysis_sample_plots(self): #analysis of minima of plots in file save for sample

        for i in range(7):

            samples_off_i=[]
            for j in range(len(self.sample_offsets)):
                soi=self.sample_offsets[j]
                samples_off_i.append(soi[i])
            samples_off_i=np.array(samples_off_i)

            self.min_sample_crb=self.crb_ii[np.where(np.abs(self.crb_ii)==np.min(np.abs(self.crb_ii)))[0][0]]
            self.min_sample_cpe=self.cpe_ii[np.where(np.abs(self.cpe_ii)==np.min(np.abs(self.cpe_ii)))[0][0]]
            self.max_sample_tso=self.tso_ii[np.where(np.abs(self.tso_ii)==np.max(np.abs(self.tso_ii)))[0][0]]
            self.min_sample_std=self.std_ii[np.where(np.abs(self.std_ii)==np.min(np.abs(self.std_ii)))[0][0]]
            self.min_sample_mse=self.mse_ii[np.where(np.abs(self.mse_ii)==np.min(np.abs(self.mse_ii)))[0][0]]
            self.min_sample_evm=self.evm_ii[np.where(np.abs(self.evm_ii)==np.min(np.abs(self.evm_ii)))[0][0]]
            self.min_offset_sample_crb=samples_off_i[np.where(np.abs(self.crb_ii)==np.min(np.abs(self.crb_ii)))[0][0]]
            self.min_offset_sample_evm=samples_off_i[np.where(np.abs(self.evm_ii)==np.min(np.abs(self.evm_ii)))[0][0]]
            self.min_offset_sample_cpe=samples_off_i[np.where(np.abs(self.cpe_ii)==np.min(np.abs(self.cpe_ii)))[0][0]]
            self.min_offset_sample_mse=samples_off_i[np.where(np.abs(self.mse_ii)==np.min(np.abs(self.mse_ii)))[0][0]]
            self.max_offset_sample_tso=samples_off_i[np.where(np.abs(self.tso_ii)==np.max(np.abs(self.tso_ii)))[0][0]]
            self.min_offset_sample_std=samples_off_i[np.where(np.abs(self.std_ii)==np.min(np.abs(self.std_ii)))[0][0]]
            self.s_evm=self.sample_vals[np.where(np.abs(self.evm_ii)==np.min(np.abs(self.evm_ii)))[0][0]]
            self.s_cpe=self.sample_vals[np.where(np.abs(self.cpe_ii)==np.min(np.abs(self.cpe_ii)))[0][0]]
            self.s_std=self.sample_vals[np.where(np.abs(self.std_ii)==np.min(np.abs(self.std_ii)))[0][0]]
            self.s_mse=self.sample_vals[np.where(np.abs(self.mse_ii)==np.min(np.abs(self.mse_ii)))[0][0]]
            self.s_tso=self.sample_vals[np.where(np.abs(self.tso_ii)==np.max(np.abs(self.tso_ii)))[0][0]]
            self.s_crb=self.sample_vals[np.where(np.abs(self.crb_ii)==np.min(np.abs(self.crb_ii)))[0][0]]

            self.save_analysis_sample_fplots(i) 

            self.min_sample_crb=-1 #reset
            self.min_sample_cpe=-1
            self.max_sample_tso=-1
            self.min_sample_std=-1
            self.min_sample_mse=-1
            self.min_sample_evm=-1
            self.min_offset_sample_crb=-1
            self.min_offset_sample_evm=-1
            self.min_offset_sample_cpe=-1
            self.min_offset_sample_mse=-1
            self.max_offset_sample_tso=-1
            self.min_offset_sample_std=-1
            self.s_evm=-1
            self.s_cpe=-1
            self.s_std=-1
            self.s_mse=-1
            self.s_tso=-1
            self.s_crb=-1

        print("\nAnalysis for Offsets based Data for all Models Done. \n")

        pass

    def analysis_offset_plots(self): #analysis fo minima of plots in file save for offset

        for i in range(7):

            self.min_vv_crb=self.crb_ii[np.where(np.abs(self.crb_ii)==np.min(np.abs(self.crb_ii)))[0][0]]
            self.min_vv_cpe=self.cpe_ii[np.where(np.abs(self.cpe_ii)==np.min(np.abs(self.cpe_ii)))[0][0]]
            self.max_vv_tso=self.tso_ii[np.where(np.abs(self.tso_ii)==np.max(np.abs(self.tso_ii)))[0][0]]
            self.min_vv_std=self.std_ii[np.where(np.abs(self.std_ii)==np.min(np.abs(self.std_ii)))[0][0]]
            self.min_vv_mse=self.mse_ii[np.where(np.abs(self.mse_ii)==np.min(np.abs(self.mse_ii)))[0][0]]
            self.min_vv_evm=self.evm_ii[np.where(np.abs(self.evm_ii)==np.min(np.abs(self.evm_ii)))[0][0]]
            self.svv_evm=self.sample_vals[np.where(np.abs(self.evm_ii)==np.min(np.abs(self.evm_ii)))[0][0]]
            self.svv_cpe=self.sample_vals[np.where(np.abs(self.cpe_ii)==np.min(np.abs(self.cpe_ii)))[0][0]]
            self.svv_std=self.sample_vals[np.where(np.abs(self.std_ii)==np.min(np.abs(self.std_ii)))[0][0]]
            self.svv_mse=self.sample_vals[np.where(np.abs(self.mse_ii)==np.min(np.abs(self.mse_ii)))[0][0]]
            self.svv_tso=self.sample_vals[np.where(np.abs(self.tso_ii)==np.max(np.abs(self.tso_ii)))[0][0]]
            self.svv_crb=self.sample_vals[np.where(np.abs(self.crb_ii)==np.min(np.abs(self.crb_ii)))[0][0]]

            self.save_analysis_offset_fplots(i) 

            self.min_vv_crb=-1 #reset
            self.min_vv_cpe=-1
            self.max_vv_tso=-1
            self.min_vv_std=-1
            self.min_vv_mse=-1
            self.min_vv_evm=-1
            self.svv_evm=-1
            self.svv_cpe=-1
            self.svv_std=-1
            self.svv_mse=-1
            self.svv_tso=-1
            self.svv_crb=-1 

        pass

    def plot_all_offsets(self, i): #plots all for changing offsets 

        self.plot_threshold_offset(i)
        self.plot_cpe_offset(i)
        self.plot_crb_offset(i)
        self.plot_std_offset(i)
        self.plot_mse_offset(i)
        self.plot_evm_offset(i)

        print("\nAll Plots for offset done, for method type {} \n".format(self.types_arr[i]))

        pass

    def plot_all_samples(self, i): #plots all for sampling exp.

        self.plot_threshold_samples(i)
        self.plot_cpe_samples(i)
        self.plot_crb_samples(i)
        self.plot_std_samples(i)
        self.plot_mse_samples(i)
        self.plot_evm_samples(i)

        print("\nAll Plots for samples done, for method type {} \n".format(self.types_arr[i]))

        pass

    def save_analysis_sample_fplots(self, sval): #save all analysis data to txt file per sample len

        filename="results/trial_{}_{}/datasave_trial_sample_{}_{}_{}.txt".format(self.stype, self.maxfilenum, self.stype, self.types_arr[sval], datetime.datetime.now())
        f=open(filename,"w")
        f.write("\nMinima Sample Analysis, for plots of type of method: {}: \n".format(self.types_arr[sval]))
        f.write("\nMinimum CRB (Cramer Rao Lower Bound): {} Hz for sample value {}, with offset value {} Hz".format(self.min_sample_crb, self.min_offset_sample_crb, self.s_crb))
        f.write("\nMinimum EVM (Error Vector Magnitude): {} dB for sample value {}, with offset value {} Hz".format(self.min_sample_evm, self.min_offset_sample_evm, self.s_evm))
        f.write("\nMinimum STD (Standard Deviation): {} Hz for sample value {}, with offset value {} Hz".format(self.min_sample_std, self.min_offset_sample_std, self.s_std))
        f.write("\nMinimum MSE (Mean Squared Error): {} Hz for sample value {}, with offset value {} Hz".format(self.min_sample_mse, self.min_offset_sample_mse, self.s_mse))
        f.write("\nMaximum Threshold (TSO): {} for sample value {}, with offset value {} Hz".format(self.max_sample_tso, self.max_offset_sample_tso, self.s_tso))
        f.write("\nMinimum CPE (Common Phase Error): {} rads for sample value {}, with offset value {} Hz".format(self.min_sample_cpe, self.min_offset_sample_cpe, self.s_cpe))
        f.close()
        
        print("\nData from Sampling Plots analysis Saved into .txt file with appropriate subdirectory successfully. For method:  {} \n".format(self.types_arr[sval]))

        pass

    def save_analysis_offset_fplots(self, sval): #save all analysis data to txt file per sample len

        filename="results/trial_{}_{}/datasave_trial_offset_{}_{}_{}.txt".format(self.stype, self.maxfilenum, self.stype, self.types_arr[sval], datetime.datetime.now())
        f=open(filename,"w")
        f.write("\nMinima Offset Analysis, for plots of type of method: {}: \n".format(self.types_arr[sval]))
        f.write("\nMinimum CRB (Cramer Rao Lower Bound): {} Hz with offset value {} Hz".format(self.min_vv_crb, self.svv_crb))
        f.write("\nMinimum EVM (Error Vector Magnitude): {} Hz with offset value {} Hz".format(self.min_vv_evm, self.svv_evm))
        f.write("\nMinimum STD (Standard Deviation): {} Hz with offset value {} Hz".format(self.min_vv_std, self.svv_std))
        f.write("\nMinimum MSE (Mean Squared Error): {} Hz with offset value {} Hz".format(self.min_vv_mse, self.svv_mse))
        f.write("\nMaximum Threshold (TSO): {} Hz with offset value {} Hz".format(self.max_vv_tso, self.svv_tso))
        f.write("\nMinimum CPE (Common Phase Error): {} Hz with offset value {} Hz".format(self.min_vv_cpe, self.svv_cpe))
        f.close()
        
        print("\nData from Offsets Plots analysis Saved into .txt file with appropriate subdirectory successfully. For method:  {} \n".format(self.types_arr[sval]))

        pass

    def save_analysis_sample(self, sval): #save all analysis data to txt file per sample len

        filename="results/trial_{}_{}/datasave_sample_analysis_trial_{}_{}_{}.txt".format(self.stype, self.maxfilenum, self.stype, sval, datetime.datetime.now())
        f=open(filename,"w")
        f.write("\n\nLogged Data From Experiment: \n\n")
        f.write("\n\nHyper-parameters considered in this experimental run (CPE correction or CPE non-correction): \n\n")
        f.write("\nSampling value (datapoints) {}".format(sval))
        f.write("\nSampling Frequency (Hz): {}\n".format(self.f_s))
        f.write("\nCarrier (Central) Frequency (Hz): {}\n".format(self.f_c))
        f.write("\nFrequency Offset to Guess (Hz): {}\n".format(self.f_t))
        f.write("\nPing Power (dB): {}\n".format(self.ping_power))
        f.write("\nNoise Power (dB): {}\n".format(self.noise_power))
        f.write("\nPilot Block Length (P): {}\n".format(self.p))
        f.write("\nPilot Block Length (M): {}\n".format(self.m))
        f.write("\nData Block Length (N): {}\n".format(self.n))
        f.write("\nNumber of Frames (B): {}\n".format(self.b))
        f.write("\nNumber of Frames (B', for Reuse): {}\n".format(self.b_dash))
        f.write("\nTransmission Block Length (Q): {}\n".format(self.q))
        f.write("\nDecimation Rate (ψ): {}\n".format(self.psi))
        f.write("\nBackoff Rate (ignored) (D): {}\n".format(self.d))
        f.write("\n\nSummary (Assumed: Bias=0, Transmitted Signal = Received Signal; for all methods): Signal Type: {}\n\n".format(self.stype))
        f.write("\nType of Analysis with Least MSE: {}, with a MSE (mean squared error) value of {} Hz, which is {} Hz for an actual offset of {} Hz\n".format(self.min_mse_type,self.min_mse_val, self.min_mse_offset, self.f_t))
        f.write("\nType of Analysis with Least STD DEV: {}, with a standard deviation value of {} Hz, which is {} Hz for an actual offset of {} Hz\n".format(self.min_std_type,self.min_std_val, self.min_std_offset, self.f_t))
        f.write("\nType of Analysis with Least Cramer Rao Lower Bound: {}, with a CRLB value of {} Hz, which is {} Hz for an actual offset of {} Hz\n".format(self.min_crb_type,self.min_crb_val, self.min_crb_offset, self.f_t))
        f.write("\nType of Analysis with Least CPE Offset: {}, with a corrected phase error value of {} rads, which is {} Hz for an actual offset of {} Hz\n".format(self.min_cpe_type,self.min_cpe_val, self.min_cpe_offset, self.f_t))
        f.write("\nType of Analysis with Least EVM value: {}, with a error vector magnitude value of {} dB, which is {} Hz for an actual offset of {} Hz\n".format(self.min_evm_type,self.min_evm_val, self.min_evm_offset, self.f_t))
        f.write("\nType of Analysis with Most Threshold value: {}, with a threshold value of {}, which is {} Hz for an actual offset of {} Hz\n".format(self.max_threshold_type,self.max_threshold_val, self.max_threshold_offset, self.f_t))
        f.write("\n\nMSE (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.mse_arr[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.mse_arr[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.mse_arr[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.mse_arr[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.mse_arr[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.mse_arr[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.mse_arr[6]))
        f.write("\n\nSTD DEV (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.std_arr[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.std_arr[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.std_arr[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.std_arr[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.std_arr[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.std_arr[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.std_arr[6]))
        f.write("\n\nCramer-Rao Lower Bound (CRLB) (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.crb_arr[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.crb_arr[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.crb_arr[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.crb_arr[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.crb_arr[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.crb_arr[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.crb_arr[6]))
        f.write("\n\nError Vector Magnitude (EVM) (dB): \n\n")
        f.write("\n{}: {} dB\n".format(self.types_arr[0], self.evm[0]))
        f.write("\n{}: {} dB\n".format(self.types_arr[1], self.evm[1]))
        f.write("\n{}: {} dB\n".format(self.types_arr[2], self.evm[2]))
        f.write("\n{}: {} dB\n".format(self.types_arr[3], self.evm[3]))
        f.write("\n{}: {} dB\n".format(self.types_arr[4], self.evm[4]))
        f.write("\n{}: {} dB\n".format(self.types_arr[5], self.evm[5]))
        f.write("\n{}: {} dB\n".format(self.types_arr[6], self.evm[6]))
        f.write("\n\nPhase Wrap Offset, Common Phase Error (CPE) (rads): \n\n")
        f.write("\n{}: {} rads\n".format(self.types_arr[0], self.cpe[0]))
        f.write("\n{}: {} rads\n".format(self.types_arr[1], self.cpe[1]))
        f.write("\n{}: {} rads\n".format(self.types_arr[2], self.cpe[2]))
        f.write("\n{}: {} rads\n".format(self.types_arr[3], self.cpe[3]))
        f.write("\n{}: {} rads\n".format(self.types_arr[4], self.cpe[4]))
        f.write("\n{}: {} rads\n".format(self.types_arr[5], self.cpe[5]))
        f.write("\n{}: {} rads\n".format(self.types_arr[6], self.cpe[6]))
        f.write("\n\nEstimated Offsets (CFO, or Carrier/Central Frequency Offsets) (Hz): \n\n")
        f.write("\n{}: {} Hz\n".format(self.types_arr[0], self.offsets[0]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[1], self.offsets[1]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[2], self.offsets[2]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[3], self.offsets[3]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[4], self.offsets[4]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[5], self.offsets[5]))
        f.write("\n{}: {} Hz\n".format(self.types_arr[6], self.offsets[6]))
        f.write("\n\nThresholds: (Ratio) \n\n")
        f.write("\n{}: {}\n".format(self.types_arr[0], self.threshold[0]))
        f.write("\n{}: {}\n".format(self.types_arr[1], self.threshold[1]))
        f.write("\n{}: {}\n".format(self.types_arr[2], self.threshold[2]))
        f.write("\n{}: {}\n".format(self.types_arr[3], self.threshold[3]))
        f.write("\n{}: {}\n".format(self.types_arr[4], self.threshold[4]))
        f.write("\n{}: {}\n".format(self.types_arr[5], self.threshold[5]))
        f.write("\n{}: {}\n".format(self.types_arr[6], self.threshold[6]))
        f.write("\n\nLegend: \n\n")
        f.write("\nAOM-NR=Angle of Mean, Non-Reuse\n")
        f.write("\nMOA-NR=Mean of Angle, Non-Reuse\n")
        f.write("\nAOM-R=Angle of Mean, Reuse\n")
        f.write("\nMOA-R=Mean of Angle, Reuse\n")
        f.write("\nMLE=Maximum Likelihood Estimate \n")
        f.write("\nBLUE=Best Linear unbiased Estimator \n")
        f.write("\n\nUser Choice List of Experiment: \n\n")
        f.write("\nSignal Type for Experiments, with U samplign for Signal Length: {}\n".format(self.signal_type_print))
        f.write("\nExperiment type: {}\n".format(self.experiment_type))
        f.close()

        print("\nData Saved into .txt file with appropriate subdirectory successfully. For sample value {} \n".format(sval))

        pass


if __name__=="__main__":

    cfo_dsp=CFO_DSP()
    stype=""
    warnings.filterwarnings("ignore")
    time.sleep(10)
    print("\nDSP CFO Est. Code Initialized... (failure to enter **SOME** advised parameters might result in runtime errors, due to inaccuracy in DSP related theories...) \n")
    user_type_exper_ch=int(input("\nEnter 0 for singular signal analysis, 1 for single signal sweep performance analysis, 2 for multi-transmitter analysis, 3 for sample sweep analysis for single signal. Enter: \n"))
    user_type_exper_ch = 0 if user_type_exper_ch < 0 else (3 if user_type_exper_ch > 3 else user_type_exper_ch) #input clipping
    if user_type_exper_ch==0:
        cfo_dsp.experiment_type="single signal optimization, 1 transmitter, default offset"
        data_control=int(input("\nEnter 0 for analysis on real data, or 1 for analysis on test data (rounding of input for all incorrect input values). Termination upon single run. Enter: \n"))
        data_control = 0 if data_control < 0 else (1 if data_control > 1 else data_control) #input clipping
        signal_i=[]
        tss=[]
        if data_control==0:
            stype="real"
            t_real, signal_real=cfo_dsp.generate_real_signal()
            _,signal_test=cfo_dsp.generate_test_signal() #only for setting hyperparams
            tss=signal_test
            signal_i=signal_real
            cfo_dsp.stype=stype
            cfo_dsp.maxfilenum=cfo_dsp.do_getmaxnum()+1
            cfo_dsp.detectdir()
            cfo_dsp.plot_real_signal(t_real, signal_real)
            print("\nFor the Real Signal...\n")
        else:
            stype="test"
            offset_choice=int(input("\nEnter 0 for keeping the default offset of 5000Hz or 1 to change it. Enter: \n"))
            offset_choice = 0 if offset_choice < 0 else (1 if offset_choice > 1 else offset_choice) #input clipping
            if offset_choice==1:
                offset_val=float(input("\nEnter the offset value, should be between {} and {} (in Hz, real value). Enter: \n".format(-cfo_dsp.f_s/2, cfo_dsp.f_s/2)))
                offset_val = -cfo_dsp.f_s/2 if offset_val < -cfo_dsp.f_s/2 else (cfo_dsp.f_s/2 if offset_val > cfo_dsp.f_s/2 else offset_val) #input clipping
                cfo_dsp.f_t=offset_val
            t_test, signal_test=cfo_dsp.generate_test_signal()
            signal_i=signal_test
            tss=signal_test
            cfo_dsp.stype=stype
            cfo_dsp.maxfilenum=cfo_dsp.do_getmaxnum()+1
            cfo_dsp.detectdir()
            cfo_dsp.plot_test_signal(t_test, signal_test)
            print("\nFor the Test Signal...\n")
        cfo_dsp.signal_type_print=stype
        tss=np.array(tss)
        signal_i=np.array(signal_i)
        
        bq_ch=int(input("\nEnter 0 for B and Q change, 1 to keep precomputed test parameters. (Advised to change). Enter: \n"))
        bq_ch = 0 if bq_ch < 0 else (1 if bq_ch > 1 else bq_ch)
        if bq_ch==0:
            cfo_dsp.precompute_bq=True
            hp_set_ch=int(input("\nEnter 0 to reset hyperparameters of m,n,q,b,b', and 1 to not do so. (Advised to do so, to prevent extremely large runtime of this code) Enter: \n"))
            hp_set_ch= 0 if hp_set_ch < 0 else (1 if hp_set_ch > 1 else hp_set_ch)
            if hp_set_ch==0:
                cfo_dsp.hardset_hyper=True
                hp_choices=int(input("\nEnter 0 to have preset values chosen, or 1 otherwise. If 0 was chosen, next 3 inputs dont matter (advised for 1). \n"))
                hp_choices= 0 if hp_choices < 0 else (1 if hp_choices > 1 else hp_choices)
                ch1=int(input("\nEnter 0,1,2,3 as indices for N values of 256,512,1024,2048 respectively (advised for 2): \n"))
                ch1= 0 if ch1 < 0 else (3 if ch1 > 3 else ch1)
                ch2=int(input("\nEnter 0,1,2,3 as indices for M values of 16,32,64,128 respectively (advised for 2): \n"))
                ch2= 0 if ch2 < 0 else (3 if ch2 > 3 else ch2)
                ch3=int(input("\nEnter 0,1,2,3 as indices for B values of 13,27,55,127 respectively (advised for 2): \n"))
                ch3= 0 if ch3 < 0 else (3 if ch3 > 3 else ch3)
                ch4=int(input("\nEnter 0 for B dash computation by sum, or 1 for B dash computation by fraction of sum (advised for 0): \n"))
                ch4= 0 if ch4 < 0 else (1 if ch4 > 1 else ch4)
                cfo_dsp.hardset_hp(hp_choices, ch1, ch2, ch3, ch4)
            else:
                cfo_dsp.hardset_hyper=False
            www=int(input("\nEnter 0,1,2,3,4 for W value (advised for 2) (combining factor) (of 2,4,6,8,16 respectively) (setter for Total Length, and Data Length) hardset. This will set B (and B') and Q again as per W constraints, after aforementioned hardset. Enter 5 to not opt in this operation (Ideally advised to not enter 5). Enter: \n"))
            www= 0 if www < 0 else (4 if www > 4 else www)
            if 0<=www<=4:
                cfo_dsp.b_effective(cfo_dsp.wu[www])
                cfo_dsp.q_effective(cfo_dsp.wu[www])
                cfo_dsp.wval_change=True
            else:
                cfo_dsp.wval_change=False
        else:
            cfo_dsp.precompute_bq=False
        nt_ch=int(input("\nEnter 0 for AWGN (Additive White Gaussian Noise), 1 for Complex 0 mean Gaussian Noise (Random), 2 otherwise. (Advised for: 2) Enter: \n"))
        nt_ch= 0 if nt_ch < 0 else (2 if nt_ch > 2 else nt_ch)
        if nt_ch==0:
            nt="awgn"
        elif nt_ch==1:
            nt="czmrgn"
        if nt_ch!=2:
            cfo_dsp.add_noise(signal_i,nt)
            cfo_dsp.noise_applied=nt
        else:
            cfo_dsp.noise_applied="N/A"
        norm_ch=int(input("\nEnter 0 for Normalizing Signal with respect to Power, 1 otherwise (advised for 1). Enter: \n"))
        norm_ch= 0 if norm_ch < 0 else (1 if norm_ch > 1 else norm_ch)
        signal_ii=[]
        if norm_ch==0:
            cfo_dsp.sig_normalized=True
            signal_ii=cfo_dsp.normalize_signal(signal_i)
        else:
            cfo_dsp.sig_normalized=False
            signal_ii=np.array(signal_i)
        chu=int(input("\nEnter 0 for Number of experiments as U, U=Signal Length, 1 for U=500 (advised for 0). Enter: \n"))
        chu= 0 if chu < 0 else (1 if chu > 1 else chu)
        if chu==0:
            cfo_dsp.utype_sample_evm="Signal Length"
        else:
            cfo_dsp.utype_sample_evm="Set to 500"
        ch_b=int(input("\nEnter (for BLUE EST.) 0 for Cramer Rao Bound (CRB) analysis by signal length method, or 1 for CRB analysis by standard deviation method (Use 1, advised). Enter: \n"))
        ch_b= 0 if ch_b < 0 else (1 if ch_b > 1 else ch_b)
        cfo_dsp.cfo_analysis(signal_ii, chu, ch_b) ##start here ... for checking whats wrong 
        cfo_dsp.save_analysis()
    elif user_type_exper_ch==1:
        cfo_dsp.experiment_type="single signal sweep efficacy, 1 transmitter, default offset"
        stype="test"
        t_test, signal_test=cfo_dsp.generate_test_signal()
        signal_i=signal_test
        tss=signal_test
        cfo_dsp.stype=stype
        cfo_dsp.pre_set_hypers()
        cfo_dsp.maxfilenum=cfo_dsp.do_getmaxnum()+1
        cfo_dsp.detectdir()
        cfo_dsp.plot_test_signal(t_test, signal_test)
        print("\nFor the Test Signal...\n")
        cfo_dsp.signal_type_print=stype
        tss=np.array(tss)
        signal_i=np.array(signal_i)
        i_v=int(input("\nEnter the increment for sweep quanta for samples: Enter: \n"))
        i_v= 5 if i_v < 5 else (len(signal_i) if i_v > len(signal_i) else i_v)
        f_i=int(input("\nEnter the factor of fraction of signal length to take for sample sorting: \n"))
        f_i = 2 if f_i <2 else (1 if f_i > 500 else f_i)
        cfo_dsp.test_by_sample(signal_i,i_v, f_i)
        for jj in range(7):
            cfo_dsp.plot_all_samples(jj)
        cfo_dsp.analysis_sample_plots()
    elif user_type_exper_ch==2:
        stype="test"
        cfo_dsp.stype=stype
        cfo_dsp.signal_type_print=stype
        cfo_dsp.experiment_type="multi transmitter analysis, single sample, choice offsets"
        t_test, _=cfo_dsp.generate_test_signal()
        cfo_dsp.pre_set_hypers()
        cfo_dsp.maxfilenum=cfo_dsp.do_getmaxnum()+1
        cfo_dsp.detectdir()
        n_transm=int(input("\nEnter number of transmitters to simulate. Enter: \n"))
        n_transm= 1 if n_transm < 1 else n_transm
        cfo_dsp.n_trans=n_transm
        for n_t in range(n_transm):
            offset_val=int(input("\nEnter the offset value for the transmitter number: {}. Enter: \n".format(n_t+1)))
            offset_val = -cfo_dsp.f_s/2 if offset_val < -cfo_dsp.f_s/2 else (cfo_dsp.f_s/2 if offset_val > cfo_dsp.f_s/2 else offset_val) #input clipping
            cfo_dsp.offset_n_trans.append(offset_val)
            cfo_dsp.inter_mc.append(1)
        for i in range(len(cfo_dsp.inter_mc)):
            cfo_dsp.mixing_coeffs.append(cfo_dsp.inter_mc)
        cfo_dsp.mixing_coeffs=np.array(cfo_dsp.mixing_coeffs)
        cfo_dsp.offset_n_trans=np.array(cfo_dsp.offset_n_trans)
        cfo_dsp.multi_signal=cfo_dsp.n_test_signals_sum(cfo_dsp.offset_n_trans, 1)
        cfo_dsp.multi_signal=np.array(cfo_dsp.multi_signal)

        m_ch=int(input("\nEnter 0 for ICA, 1 for NMF. Enter: \n"))
        m_ch= 0 if m_ch < 0 else (1 if m_ch >1 else m_ch)
        if m_ch==0:
            cfo_dsp.nested_signals=cfo_dsp.ica(cfo_dsp.multi_signal,2)
        elif m_ch==1:
            cfo_dsp.nested_signals=cfo_dsp.nmf(cfo_dsp.multi_signal,2)
        cfo_dsp.nested_signals=np.array(cfo_dsp.nested_signals)
        for i in range(len(cfo_dsp.nested_signals)):
            signal_test=cfo_dsp.nested_signals[i]
            cfo_dsp.plot_test_signal(t_test, signal_test)
            print("\nFor the Test Signal... at offset {}\n".format(cfo_dsp.offset_n_trans[i]))
            tss=np.array(tss)
            signal_i=np.array(signal_i)
            cfo_dsp.cfo_analysis(signal_i, 0, 1)
            cfo_dsp.save_analysis()
    elif user_type_exper_ch==3:
        cfo_dsp.experiment_type="offset sweep analysis, 1 transmitter, 1 sample"
        stype="test"
        cfo_dsp.stype=stype
        cfo_dsp.maxfilenum=cfo_dsp.do_getmaxnum()+1
        cfo_dsp.signal_type_print=stype
        cfo_dsp.detectdir()
        i_v=int(input("\nEnter the increment for offset quanta in Hz: Enter: \n"))
        i_v= 5 if i_v < 5 else (500 if i_v > 500 else i_v) #hard limit on max 
        f_i=int(input("\nEnter the factor of fraction of signal length to take for sample sorting: \n"))
        f_i = 2 if f_i <2 else (1 if f_i > 500 else f_i)
        for off in range(int(-cfo_dsp.f_s/(2*f_i)), int(1+(cfo_dsp.f_s/(2*f_i))), i_v):
            cfo_dsp.f_t=off
            t_test, signal_test=cfo_dsp.generate_test_signal()
            cfo_dsp.pre_set_hypers()
            signal_i=signal_test
            tss=signal_test
            cfo_dsp.plot_test_signal(t_test, signal_test)
            print("\nFor the Test Signal... at offset {}\n".format(off))
            tss=np.array(tss)
            signal_i=np.array(signal_i)
            cfo_dsp.cfo_analysis(signal_i, 0, 1)
            cfo_dsp.save_analysis()
            cfo_dsp.offset_vals.append(off)
            cfo_dsp.mse_arr=[] #reset
            cfo_dsp.std_arr=[]
            cfo_dsp.crb_arr=[]
            cfo_dsp.offsets=[]
            cfo_dsp.cpe=[]
            cfo_dsp.evm=[]
            cfo_dsp.threshold=[]
        cfo_dsp.offset_vals=np.array(cfo_dsp.offset_vals)
        for jj in range(7):
            cfo_dsp.plot_all_offsets(jj)
        cfo_dsp.analysis_offset_plots()
    print("\nCode Instance successfully exited. \n") #idea is that: samples are set as per best after checking which one, and the efficacy for discerning offsets for multiple sources and then separately also for multiple offsets, is seen

#% KEEP CURSOR HERE FOR IDE RUNS ---> () #%
