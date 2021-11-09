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

print('================ NON-RESAMPLED DICOM DIMENSIONS ============')
print('Image size: ' + str(DICOM.GetSize()))
print('Image spacing: ' + str(DICOM.GetSpacing()))


print('================ RESAMPLED DICOM DIMENSIONS ================')
print('Image size: ' + str(DICOM_resampled.GetSize()))
print('Image spacing: ' + str(DICOM_resampled.GetSpacing()))

#===============================================================================================

#=========================== RESAMPLING THE MASK ===============================================

rtstruct = RTStructBuilder.create_from(
  dicom_series_path="/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT", 
  rt_struct_path="/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT/3-2.dcm"
)

# Getting arrays for all the masks for the determined ROIs
mask_3d_Lung_Right = rtstruct.get_roi_mask_by_name("Lung-Right") 
mask_3d_Lung_Left = rtstruct.get_roi_mask_by_name("Lung-Left")
mask_3d_GTV_1 = rtstruct.get_roi_mask_by_name("GTV-1")
mask_3d_spinal_cord = rtstruct.get_roi_mask_by_name("Spinal-Cord")

# Setting what the desired mask is (for the case of a tumour we out GTV-1)
mask_3d = mask_3d_GTV_1

#Converting this array from boolean to binary
mask_3d = mask_3d + 1
mask_3d = mask_3d - 1

mask_3d_image = sitk.GetImageFromArray(mask_3d)

print('================ NON-RESAMPLED RTSTRUCT DIMENSIONS ==========')
print('Image size: ' + str(mask_3d_image.GetSize()))
print('Image spacing: ' + str(mask_3d_image.GetSpacing()))

mask_3d_image_resampled = resample_volume(mask_3d_image)

print('================ RESAMPLED RTSTRUCT DIMENSIONS ==============')
print('Image size: ' + str(mask_3d_image_resampled.GetSize()))
print('Image spacing: ' + str(mask_3d_image_resampled.GetSpacing()))



