# DSP-PING

A module to improve ping detection and filtering techniques for the radio telemetry project at UCSD Department of EE

# Specific Features:

~Uses CFO (Carrier Frequency Offset) Estimation techniques as novel research guidepoints to decode the offset in transmission for the wideband signal received and further processed via AC (Auto Correlation) filter functions. 

## Mathematics of Signal Processing Utilized: 

~Methods of Angle Of Mean, Mean of Angle, with Non/Reuse Data Schemes, MLE (Maximum Likelihood Estimates), BLUE (Best Linear Unbiased Estimator) and conventional are used, and evaluated by EVM (Error Vector Magnitude), CRLB (Cramer Rao Lower Bound) and other basic statistical metrics (such as standard deviation and so on). 

~These methods effectively show an ecnapsulated view of the accuracy of all methods, with respect to testable real and simulated signal data.

~Detects ping via first finalizing the transmitted offset from raw data of signal that is processed through AC and then by multiple estimators, which are then evaluated for accuracy

~Neural Network or Matrix based solutions exist, but wer ento implemented to save processing complexity for other tasks on the drone chip, and to save time complexity with DSP based tasks. 

## Further Optimization:

~Sweep analysis of sampling size as per all 7 model metrics and threshold of 200Hz in absolute frequency found

~Efficacy of model across different offsets encountered checked

~Models tested against sample data from 2021, test data generating, and HACK RF triple DSR data from 2022

~Models tested fro non-amplitude specific ICA and NMF vectorizations for multi-transmitter filtering and offset+sample optimizations in parallel methodologies. 

# Dataset:

Download the dataset in: https://drive.google.com/drive/folders/18TVyMU-xf88z0sr3A5C-CD5P6WQs4LhZ starting from the name: "RAW_DATA_...". make a folder named "datafiles" and insert this file in the new folder. Then this "datafiles" folder is inserted into the "dsp" superfolder. 

Since the original real dataset is too large to upload here, it is maintained on a google drive, which will need the afoermentioneed manual steps for real analysis on your local machine. More files can be similarly added in the future if needed, and modifications to code to account for analysis of multiple real signal binaries will be done then. 

# Results:

The results will automatically be stored in the "results" subdirectory, where the trial number of the user initiated run and the test/real signal type will be the subfolder name. The folder containes a .txt file of all results, user inputs, abbreviations, explanations and metrics, and a .png file of the real/test signal plot for reference.

# Code Specs:

~Written entirely in Python 3.8+

~Uses numpy, scipy, os, warnings, datetime, time, struct matplotlib libraries extensively

~This is for mathematical analysis, mathematical calculations, minor resolution handling, filepath traversal, signal plotting, and data reading purposes only. Latest versions of all of them via a pip install command in the terminal will work well. 

# UCSD E4E Github:

This repository has the following relation to the UCSD RCT E4E project repository: 

## RCT Main Branch and NH0 Branch in new DSP2 Repo: 

Merge as feature to tree of rct-signalprocessing for NH0 branch within RCT main for UCSD RCT Project.

Main Repo Link: https://github.com/UCSD-E4E/rct_signal_characterization

NH0 Branch Link: https://github.com/UCSD-E4E/rct_signal_characterization/tree/NH0

New DSP2 Repo, upon which the main and NHO is rebuilt: https://github.com/UCSD-E4E/radio_collar_tracker_dsp2

# Ownership:

Owned by Nathan Hui, Melina Dimitropoulou Kapsogeorgou, Arya Keni, Department of Electrical Engineering at the University of California San Diego. 

# © University of California, San Diego (Computer Science and Engineering Department)
# © Beckman Institute of Conservation Research (BICR) in the San Diego Zoo
