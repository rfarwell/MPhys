"""
This program will read in the clinical data csv file, convert it to an array and then produce a histogram 
of the timepoints at which people have died.
"""
#============================================================================================================================================
#=========================== IMPORTING FUNCTIONS ============================================================================================
#============================================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
#============================================================================================================================================
#============================================================================================================================================
#============================================================================================================================================

#============================================================================================================================================
#=========================== DEFINING FUNCTIONS =============================================================================================
#============================================================================================================================================
def get_clinical_data():
    """
    This function retrieves the clinical data file and converts to one big array containing all the data.
    Two new arrays are then made containing the timepoints and dead status, respectively.
    These are then combined into one 'data' array which is returned.

    No inputs are required since the filepath is predefined in the code

    Rory Farwell 30/01/2022
    """
    clinical_data = np.genfromtxt(clinical_data_filepath,comments = '%', delimiter = ',') # Opening the clinical cata csv file and forming array from the data.

    # Assign new array from useful information in the clinical data
    timepoints = clinical_data[:,8]
    dead_statuses = clinical_data[:,9]
    return timepoints, dead_statuses

def total_dead() :
    """
    This function counts the number of patients in the clinical data that are dead (regardless of last follow up time).

    Rory Farwell 30/01/2022
    """
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
    
    # print(f"From this data set {dead_counter} are dead.")

    return dead_time, alive_time

def non_cumulative_hist() :
    """
    This function plots a histogram of the timepoints at which people have died, using the dead_time array defined in the total_dead function.

    Rory Farwell 30/01/2022
    """

    plt.hist(dead_time, bins = 100)
    plt.xlabel("Time of death (days)")
    plt.ylabel("Frequency of Ocurrence")
    
    return plt.show()

def cumulative_hist() :
    """
    This function plots a cumulative histogram of the timepoints at which people have died, using the dead_time array defined in the total_dead function.

    Rory Farwell 30/01/2022
    """
    plt.hist(dead_time, bins = 100, cumulative = True, histtype='step')
    plt.xlabel("Time of death (days")
    plt.ylabel("Cumulative frequency")

    return plt.show()

def threshold_time_calculations(threshold_time) :
    """
    This function takes an input of the desired threshold day. It outputs values for the number of patients who are alive and dead on the threshold day.
    It also indicates the number of patients whose data is right-censored at this time.

    Rory Farwell 30/01/2022 
    """
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
    print(f"Dead after {threshold_time} days: {threshold_dead_counter}")
    print(f"Alive after {threshold_time} days: {threshold_alive_counter}")
    print(f"No info obtained from {threshold_no_info_counter} patients")
    ratio_of_dead_to_alive = threshold_dead_counter/threshold_alive_counter
    print(f"Dead to alive ratio at a threshold time of {threshold_time} = {ratio_of_dead_to_alive}")
    return 
#============================================================================================================================================
#============================================================================================================================================
#============================================================================================================================================

#============================================================================================================================================
#=========================== FINAL CODE =====================================================================================================
#============================================================================================================================================
clinical_data_filepath = "/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC-Radiomics-Clinical-Data.csv" #setting file path to the clinical data csv file
threshold_day = 547.5 # defining the threshold day/

timepoints, dead_statuses = get_clinical_data() # Array definiton.
dead_time, alive_time = total_dead() # Array definiton.

non_cumulative_hist() # Plot histogram of when patients, who are dead, died.
cumulative_hist() # Plot cumulative histogram of when patients, who are dead, died.

threshold_time_calculations(threshold_day) # Perform threshold day calculations and print the results.
#============================================================================================================================================
#============================================================================================================================================
#============================================================================================================================================
