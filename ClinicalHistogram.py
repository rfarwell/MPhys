"""
This program will read in the clinical data csv file, convert it to an array and then produce a histogram 
of the timepoints at which people have died
"""
import numpy as np
import matplotlib.pyplot as plt

clinical_data_filepath = "/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC-Radiomics-Clinical-Data.csv" #setting file path to the clinical data csv file


clinical_data = np.genfromtxt(clinical_data_filepath,comments = '%', delimiter = ',') #opening the csv file

timepoints = clinical_data[:,8]
#print(timepoints)

dead_statuses = clinical_data[:,9]

#print(dead_statuses.shape)
data = np.empty((422,2), dtype = float)

print(data)
for i in range(len(timepoints)) :
    data[i][0] = timepoints[i]
    data[i][1] = dead_statuses[i]

data = data.astype(int)
print(data)

dead_time = []

for i in range(len(timepoints)) :
    if dead_statuses[i] == 1 :
        dead_time.append(timepoints[i])

plt.hist(dead_time, bins = 100)
plt.xlabel("Time of death (days)")
plt.ylabel("Frequency of Ocurrence")
plt.show()