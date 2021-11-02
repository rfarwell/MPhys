"""

This code will provide more information from the DICOM data series.
It takes into account all of the DICOM series in the selected directory
and outputs the number of rows in an image, the number of columns in an image,
the slice spacing in an image and the number of images in the series.

The row and column information will be used to check the 2D dimensions of the files.
The slice spacing information will be used for voxel resizing (preprocessing).
The number of slices/images in the series will be used to ensure all series have the same
number of images because a CNN expects all inputs to have predefined dimensions.

Rory Farwell (26/10/2021)

"""
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import sys
import os
reader = sitk.ImageSeriesReader()
# Note : It is currently specified for our data but in the future may be made more general

counter = 0 #Used to check that the code is running

#Defining the start of the path name to where my files are (this depends on your path and the
# naming scheme you used when using dicomsort.py)
directory = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-'

#defining the number of iterations the for loop will perform. +1 is due to the range function
number_of_iterations = 5+1

biggest_size = 0
biggest_size_label = 0
rows = []
columns = []
depth=[]
num_imgs = []
size = []

for x in range(1, number_of_iterations) :
    _3_digit_x = '{0:03}'.format(x) #formats the 'x' to the form 'yzx' e.g. 1 -> 001
                                    # so that it fits the format of the naming scheme used
                                    # e.g. LUNG1-001-CTUnknownStudyID
    directory_full = directory + str(_3_digit_x) + '-CTUnknownStudyID/' #   This line will change depending on the naming scheme that you have used
    dicom_names = reader.GetGDCMSeriesFileNames(directory_full)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    rows.append(size[0])
    columns.append(size[1])
    spacing = image.GetSpacing()
    depth.append(spacing[0])
    num_imgs.append(size[2])
    #print(num_imgs[x-1])
    #print(spacing)
    counter += 1
    print(counter) # Checking that the code is running and its progress

largest_number_of_series = np.max(num_imgs)
series_number_with_largest_number_of_series = np.argmax(num_imgs)

#Prints the maximum number of slices in a series and tells you which series this is.
print("The largest depth in the NSCLC-Radiomics data set is " + str(np.max(largest_number_of_series)) + " from LUNG1-" + str('{0:03}'.format(series_number_with_largest_number_of_series)))
print("The largest number of rows in the data is " + str(np.max(rows)) + " and the minimum number of rows is " + str(np.min(rows)) + ". If these are the same then all have the same number of rows.")
print("The largest number of columns in the data is " + str(np.max(columns)) + " and the minimum number of rows is " + str(np.min(columns)) + ". If these are the same then all have the same number of columns.")
#print("The largest spacing in the data is " + str(np.max(spacing)) + " and the minimum spacing is " + str(np.min(spacing)) + ". If these are the same then all have the same spacing.")
print("The largest depth in the data is " + str(np.max(depth)) + " and the minimum depth is " + str(np.min(depth)) + ". If these are the same then all have the same spacing.")

"""
# The below prints all of the data collected
print("Rows:")
print(rows)
print("~~~~~~~~~~~~~~~")
print("Columns:")
print(columns)
print("~~~~~~~~~~~~~~~")
print("Spacing:")
print(spacing)
print("~~~~~~~~~~~~~~~")
print("Depth:")
print(depth)
print("~~~~~~~~~~~~~~~")
print("Num_imgs:")
print(num_imgs)
"""

"""
reader = sitk.ImageSeriesReader()
dcm_paths =reader.GetGDCMSeriesFileNames(root)

dcm_names = sorted([path for path in dcm_paths])

resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())



    """
# def WL_norm(img, window=1000, level=700):
#     """
#     Apply window and level to image
#     """

#     maxval = level + window/2
#     minval = level - window/2
#     wl = sitk.IntensityWindowingImageFilter()
#     wl.SetWindowMaximum(maxval)
#     wl.SetWindowMinimum(minval)
#     out = wl.Execute(img)
#     return out    