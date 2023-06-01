import numpy as np
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
import warnings
import os
import datetime
import time
from pathlib import Path
from smb_unzip.smb_unzip import smb_unzip #be sure to follow instructions: https://github.com/UCSD-E4E/smb-unzip 


class CFO_DSP:

    #INIT PARAMS: 

    def __init__(self):# Generated signal parameters

        self.t_end = 4 # s
        self.f_s = 1000000 # Hz #t_s for psi
        self.f_c = 172000000 # Hz
        self.f_t = 5000 # Hz offset from center to guess
        self.t_ping = 0.05 # s
        self.ping_period = 1 # s
        self.ping_power = -96 # dB
        self.noise_power = -60 # dB
        self.dataFilePath="datafiles/RAW_DATA_000002_000001" #real load 
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
        self.min_mse_offset=-1
        self.min_crb_offset=-1
        self.min_evm_offset=-1
        self.min_cpe_offset=-1
        self.min_std_offset=-1
        self.maxfilenum=-1

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

        self.mse_arr=np.array(self.mse_arr)
        self.std_arr=np.array(self.std_arr)
        self.crb_arr=np.array(self.crb_arr)
        self.types_arr=np.array(self.types_arr)
        self.offsets=np.array(self.offsets) #normalized offset per generic smapling formula can be calculated but was ignored
        self.cpe=np.array(self.cpe) #CPE per generic formula for sampling period can be computed but is ignored. 
        self.evm=np.array(self.evm) #IT IS POSSIBLE TO ADD ABOVE IN A LOOP OF len(offsets) AND ADD CONDITIONALS (if i==1) TO DISCERN SLIGHT DIFFERENCES IN METHODICAL EVALUATIONS

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

        self.min_mse_offset=self.offsets[np.where(np.abs(self.mse_arr)==np.min(np.abs(self.mse_arr)))[0][0]]
        self.min_crb_offset=self.offsets[np.where(np.abs(self.crb_arr)==np.min(np.abs(self.crb_arr)))[0][0]]
        self.min_evm_offset=self.offsets[np.where(np.abs(self.evm)==np.min(np.abs(self.evm)))[0][0]]
        self.min_cpe_offset=self.offsets[np.where(np.abs(self.cpe)==np.min(np.abs(self.cpe)))[0][0]]
        self.min_std_offset=self.offsets[np.where(np.abs(self.std_arr)==np.min(np.abs(self.std_arr)))[0][0]]

        print("\nAnalysis Complete. Now this data can be used for FFT or Autocorrelation + RSSI based Ping Detection.\n")

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

if __name__=="__main__":

    cfo_dsp=CFO_DSP()
    stype=""
    warnings.filterwarnings("ignore")
    time.sleep(10)
    print("\nDSP CFO Est. Code Initialized... \n")
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
    cfo_dsp.cfo_analysis(signal_ii, chu, ch_b) #analyze non CPE first, save
    cfo_dsp.save_analysis()
    print("\nCode Instance successfully exited. \n")

#% KEEP CURSOR HERE FOR IDE RUNS ---> () #%