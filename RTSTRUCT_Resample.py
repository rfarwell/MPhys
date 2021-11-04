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

counter = 0

def resample_volume(volume, interpolator = sitk.sitkNearestNeighbor):
    """
    Using the Nearest Nehgbour interpolator to keep clean edges on the mask
    """
    new_size = [512, 512, 256]
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([1, 1, 4])
    resample.SetDefaultPixelValue(-1024)

    return resample.Execute(volume)

directory = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-'