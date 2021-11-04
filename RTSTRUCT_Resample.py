"""
This code will try to resample a single RTSTRUCT file to the same
dimensions of the DICOM series
"""

from __future__ import print_function
import SimpleITK as sitk
import numpy as np #not sure whether this is needed
import scipy
from scipy import stats #not sure whether this is needed
import os
reader = sitk.ImageSeriesReader()

#===========================ATTEMPT 1===============================================
#========UNSUCCESSFUL - FILE NAMES INFORMATION IS EMPTY. CANNOT READ SERIES=========
# counter = 0

# def resample_volume(volume, interpolator = sitk.sitkNearestNeighbor):
#     """
#     Using the Nearest Nehgbour interpolator to keep clean edges on the mask
#     """
#     new_size = [512, 512, 256]
#     resample = sitk.ResampleImageFilter()
#     resample.SetInterpolator(interpolator)
#     resample.SetOutputDirection(volume.GetDirection())
#     resample.SetOutputOrigin(volume.GetOrigin())
#     resample.SetSize(new_size)
#     resample.SetOutputSpacing([1, 1, 4])
#     resample.SetDefaultPixelValue(-1024)

#     return resample.Execute(volume)

# directory = '/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT/3-2.dcm'
# output_file = '/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT_resampled/3-2.dcm'


# reader = sitk.ImageSeriesReader()
# rtstruct_path = reader.GetGDCMSeriesFileNames('/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT/3-2.dcm')
# reader.SetFileNames(rtstruct_path)
# volume = sitk.ReadImage(rtstruct_path)
# x = resample_volume(volume)
# sitk.WriteImage(x, output_file)
#=======================================================================================

#===========================ATTEMPT 2===============================================
"""
In this attempt, I will not attempt to resample the RTSTRUCT fie but will try to resample
the array produced, by the rt_utils library, from the RTSTRUCT file
"""
