# dsp-rct-multi-team
DSP RCT Project from E4E of ECE Department of UCSD, Jacobs School of Engineering

## Helpful Links and Sources for Relevant Implementations

The Repository which this one ties into: https://github.com/UCSD-E4E/rct-dsp3 (official RCT DSP from E4E)

References of CodeBases: (from RCT, UCSD, ECE):

RCT Postprocessing: https://github.com/UCSD-E4E/radio_collar_tracker_postprocess

RCT DSP official Drone Version 1: https://github.com/UCSD-E4E/radio_collar_tracker_dsp

RCT Signal Characterization Theory: https://github.com/UCSD-E4E/rct_signal_characterization

RCT DSP official Drone Version 2: https://github.com/UCSD-E4E/radio_collar_tracker_dsp2


References for BLUE as a concept and implementations:

(1): BLUE-py: https://github.com/jolange/BLUE-py

(2): blue: https://github.com/tomneep/blue

(3): PyBLUE: https://github.com/rdisipio/PyBLUE

(4): BLUEST: https://github.com/croci/bluest




For NAS access: to gain direct RAW directory access: contact Hannah Grehm (UCSD) or Nathan Hui (UCSD) at their _email.ucsd.edu_ institutional emails. 

Use the repository (clone, download, install) for imports via JSON format directly from NAS into .py files for RAW data: https://github.com/UCSD-E4E/smb-unzip

Team: Arya Keni (Penn State Univ.), Haochen Wang (UCSD)



## About:

This DSP module is existant to aid in solving multi transmitter and single receiver RF separation via CFO estimation in SDR domains. 

Here, **references_v1** contains key papers regarding the theory, mathematics, and analysis of CFO analysis.

The other directory (with DSP in the title) contains presentation slides and their pdf versions for single CFO estimation along with its extensive analysis and theory. 

The file **time_freq_domain_sep_fns.pdf** contains analysis on signal generation of multi transmitter conditions. 

The file **test_numfind_ece.py** contains data on testing of certain convolution and autocorrelation methods that are custom marked for wireless transmission. 

Check **requirements.txt** for information on needed key libraries. 

The code for multi signal generation in real conditions is in **prez_ftser_sigs_multi.py** and results for this code are in **outputs_ftdomain**. This is all in the **spectral folder**. Nothing else in the folder is relevant.

The folders **signal_variations** and **ftd** is irrelevant.

Within **dsp**, check **offset_est.py** for the main single transmitter CFO estimation analysis and code, estimated within bounds of 7 methods. Its analysis is in **results** as files. Nothing else in that folder matters. 

In **dsp_spring2023**, the .py file contains methods to convert RAW files from SDR sources to a combined multi signal, as well as extract form binary to complex on conformal occasions. The generated signal for complex and single domains exists in its own analytical .py files as mentioned above. Additionally, **RAW_DATA_parser.py** in this folder contains crucial methodology on SDR data from drones to be analyzed in a DSP aspect by converting signals received for post-processing. 

The latest adaptations on multi-CFOs are in the **dsp_summer2023** folder, where the **outputs_ftdomain** folder contains hyperparametric analysis on 

For more information on CFO, get started at: https://en.wikipedia.org/wiki/Carrier_frequency_offset

## Current Capabilities: 

~Single CFO estimation from a carrier frequency in the sub-GHz range (1-3GHz), for about 50% skew in either direction for the offset from source, with 7 methods to analyze the efficacies over range of test cases, including SNR, ping power, and delays. 

~Mathematical analysis on estimation methods via MSE, MAE, MPE, and textfile printing of analytical estimates to a true vector. 

~Signal Generation in Complex Domain for Single and Multi Strands from DSR, theoretically, and with SDR captured RAW binary samples.

~Bitstream conversion and dense data handling with CFO analysis of real data. 

~Signal conversion to FT domain from bitstream, with subsequent proof of concept

~FFT to IFFT to FFT of signals for visualization in frequency and/or time domains. Visualization plots in other complex fields available (STFT, etc.)

~Data is one which accounts for manipulation of real data to deadlock and deadtime, along with amplitude obfuscation regimes. 

~Handling server zips of large real data readings, along with management of local hierarchies. 

## Future Directions:

~Ascertain a clear method for single CFO estimation based on analysis, and test out the details. 

~Ascertain a clear method for multi CFO estimation based on analysis, and test out the details. 

~Adapt to a more robust RAW data scheme, for appropriate verification of CFO existence (for true markers of CFO to be compared)

~Optimize the multi CFO for hyperparametric coherence. 

~Rigorous tesbenching on multiple domains, efficacy measures, transforms, and limitations of extrinsic properties of SDRs and environmental situations, with analytical plots and measurable solutions. 

~Other signal decomposition imaging techniques, such as JFT, CFT, etc. 
