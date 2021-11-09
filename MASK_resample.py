"""
This code will use the ResampleImageFilter function to resample both the DICOM and Mask for a relevant DICOM series
"""

#========================== IMPORTING LIBRARIES =================================================
import rt_utils
import sys
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy
from scipy import stats
import os
reader = sitk.ImageSeriesReader()
#================================================================================================

#========================== RESAMPLING THE DICOM FILE ===========================================
def resample_volume(volume, interpolator = sitk.sitkLinear) :
    '''
    This function resample a volume to size 512 x 512 x 256 with spacing 1 x 1 x 4 (Good for our dataset)
    '''
    new_size = [512, 512, 256]
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([1, 1, 4])
    resample.SetDefaultPixelValue(-1024)

    return resample.Execute(volume)


# Directing the code for where to look for the DICOM series and where to output this file to.

filepath = '/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT'

reader = sitk.ImageSeriesReader()
dcm_paths = reader.GetGDCMSeriesFileNames(filepath)
reader.SetFileNames(dcm_paths)
DICOM = sitk.ReadImage(dcm_paths)
DICOM_resampled = resample_volume(DICOM)

print('Image size: ' + str(DICOM_resampled.GetSize()))