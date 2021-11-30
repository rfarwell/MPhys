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

#print(data)
for i in range(len(timepoints)) :
    data[i][0] = timepoints[i]
    data[i][1] = dead_statuses[i]

data = data.astype(int)
#print(data)

dead_time = []
dead_counter = 0 #counting how many people are dead

for i in range(len(timepoints)) :
    if dead_statuses[i] == 1 :
        dead_counter += 1
        dead_time.append(timepoints[i])

alive_time = []
for i in range(len(timepoints)) :
    if dead_statuses[i] == 0 :
        alive_time.append(timepoints[i])

print(f"From this data set {dead_counter} are dead.")


# plt.hist(dead_time, bins = 100)
# plt.xlabel("Time of death (days)")
# plt.ylabel("Frequency of Ocurrence")
# plt.show()

    
# plt.hist(dead_time, bins = 100, cumulative = True, histtype='step')
# plt.xlabel("Time of death (days")
# plt.ylabel("Cumulative frequency")
# plt.show()
dead_time = np.array(dead_time)
alive_time = np.array(alive_time)
# print(len(dead_time))
# print(len(alive_time))

threshold_time = 547.5
threshold_dead_counter = 0
threshold_alive_counter = 0
threshold_no_info_counter = 0

for i in range(len(dead_time)) :
    if dead_time[i] < threshold_time :
        threshold_dead_counter += 1
    elif dead_time[i] >= threshold_time :
        threshold_alive_counter += 1

for i in range(len(alive_time)) :
    if alive_time[i] < threshold_time :
        threshold_no_info_counter += 1
        print(f"No info obtainable from INDEX {i}")
    elif alive_time[i] >= threshold_time :
        threshold_alive_counter += 1
print("======================")
print(f"Dead after 1.5 years: {threshold_dead_counter}")
print(f"Alive after 1.5 years: {threshold_alive_counter}")
print(f"No info obtained from {threshold_no_info_counter} patients")
ratio_of_dead_to_alive = threshold_dead_counter/threshold_alive_counter
print(f"Dead to alive ratio at a threshold time of {threshold_time} = {ratio_of_dead_to_alive}")
