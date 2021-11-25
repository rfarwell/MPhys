
"""
This code tells you which DICOM series in your directory has the 
largest number of images in it. This information will be used to determine
how many slices of 'air' need to be added to each DICOM series because
the CNN used later will expect all of the input data to have the same dimensions

Rory Farwell (26/10/2021)
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~IMPORTING FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function
import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
reader = sitk.ImageSeriesReader()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEFINING VARAIABLES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

counter = 0 #Used to check that the code is running

"""Defining the start of the path name to where my files are (this depends on your path and the
   naming scheme you used when using dicomsort.py)"""
directory = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-'



"""defining the number of iterations the for loop will perform. +1 is due to the range function"""
number_of_iterations = 422+1 #I want to make this so that the program can read the number of series in the chosen directory

biggest_size = 0
biggest_size_label = 0
sizes = []
too_big_sizes = []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for x in range(1, number_of_iterations) :
      _3_digit_x = '{0:03}'.format(x) #formats the 'x' to the form 'yzx' e.g. 1 -> 001
                                    # so that it fits the format of the naming scheme used
                                    # e.g. LUNG1-001-CTUnknownStudyID
      directory_full = directory + str(_3_digit_x) + '-CT' #   This line will change depending on the naming scheme that you have used
      dicom_names = reader.GetGDCMSeriesFileNames(directory_full)
      reader.SetFileNames(dicom_names)
      image = reader.Execute()
      size = image.GetSize()
      sizes.append(size[2])
      
      if size[2] > 170 :
         too_big_sizes.append(size[2])

      if size[2] > biggest_size :
         biggest_size = size[2]
         biggest_size_label = x
      counter += 1
      print(counter) # Checking that the code is running and its progress
      
print("The largest depth in the NSCLC-Radiomics data set is " + str(biggest_size) + " from LUNG1-" + str('{0:03}'.format(biggest_size_label)))

# print(sizes)
print(too_big_sizes)
print(len(too_big_sizes))
