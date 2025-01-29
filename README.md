  This repository provides a brief description of the  Health-Spa dataset alongside an example python file with preprocessing and technical validation steps.

  The dataset combines signals from two domains: ambient environmental conditions (e.g., temperature, humidity, light, sound, air quality) 
and physiological responses (e.g., cardiac and electrodermal activity). Data was collected individually over approximately 48 minutes per participant, 
totaling 600 minutes, under varied indoor conditions. Ambient signals and electrodermal activity were sampled at 1 Hz, resulting in over 30,000 data points. 
Cardiac activity and sound were sampled at higher frequencies, generating significantly more data points. Machine Learning models were developed using this 
dataset to classify ambient environmental risks to cardiac health and estimate physiological responses based solely on environmental data. The dataset enables 
exploration of the links between indoor climates and health, paving the way for advancements in context-aware technology.


  File naming convention is designed to convey the type of data, the participant and the data collection scenario associated with each file. For example sound 
data for participant number one, for scenario number one is stored in a file named "sound\_P1\_S1.csv". 

  The file ambient\_Pn\_Sn.csv comprises sixteen columns, encapsulating all the environmental parameters except sound and also storing measurements 
for electrodermal activity. The first column in the file corresponds to skin resistance measurements, while the second column denotes light intensity 
measurements in lux. Subsequent columns three through five provide data on PM1, PM2.5, and PM10 concentrations, respectively. Columns six through eleven 
represent measurements of the number of dust particles with diameters exceeding 0.3, 0.5, 1, 2.5, 5, and 10 micrometers, respectively. Column twelve is 
designated for temperature readings, column thirteen for humidity, column fourteen for pressure, and column fifteen for Indoor Air Quality (IAQ). 
Finally, the last column records the timestamp of each sample.

  The file sound\_Pn\_Sn.csv includes a header containing information about sampling frequency, blocksize, windowing, weighting, averaging, minimum frequency, 
and maximum frequency. The data within comprises two columns: the first column denotes time, while the second column records sound levels in decibels.

  The rtor\_Pn\_Sn.csv file has two columns, the first one indicates raw R-to-R interval values and the second indicates the corrected time, in seconds, 
at which the R-peaks occurs. 

  The file ecg\_Pn\_Sn.csv consists of seven columns. The first column is allocated for time, the second column for ECG\_DATA, 
containing voltage data samples of ECG, and the third column for ECG Data Tags (ETAG). The fourth column is designated for Pace Data Tags (PTAG), while the 
fifth and sixth columns represent ECG measurements in millivolts before and after filtering, respectively. The last column records the type of filter applied. 
ETAG values can assume either 000 or 010, denoting that the ECG data for the sample encompasses both valid voltage and time steps within the ECG record. 
Conversely, values of 001 and 011 indicate transient and invalid voltage information, while the time step remains valid. PTAG values range from 000 to 101 
if pace events are detected, while a value of 111 denotes the absence of pace events during the sample recording.

  An aggregated dataset file for each participant, Final\_Dataframe\_Pn.csv, has been compiled for convenience. It comprises eleven columns extracted and 
concatenated from the raw data files, encompassing all ambient environmental parameters, as well as the two physiological indicators.
