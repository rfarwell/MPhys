"""
This code will use the ResampleImageFilter function to resample both the DICOM and Mask for a relevant DICOM series
"""

#========================== IMPORTING LIBRARIES =================================================
from genericpath import getsize
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

#========================== DEFINING RESAMPLING VARIABLES =======================================
Output_Spacing = [1, 1, 1]
new_size = [512, 512, 512]
#================================================================================================

#========================== DEFINING FUNCTIONS ==================================================

def resample_DICOM(volume, interpolator = sitk.sitkLinear) :
    '''
    This function resample a volume to size 512 x 512 x 512 with spacing defined from Output_Spacing
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(-1024)

    return resample.Execute(volume)


def resample_MASK(volume, interpolator = sitk.sitkNearestNeighbor) :
    '''
    This function resample a volume to size 512 x 512 x 256 with spacing 1 x 1 x 4 (Good for our dataset)
    '''
    print(volume.GetSize())
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    #resample.SetOutputSpacing([Output_Spacing[2], Output_Spacing[1], Output_Spacing[0]])
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(0)

    return resample.Execute(volume)


def permute_axes(volume, permutation_order) :
    """
    This function permutes the axes of the input volume.
    """
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder(permutation_order)

    return permute.Execute(volume)


#================================================================================================

#========================== RESAMPLING THE DICOM FILE ===========================================
# Directing the code for where to look for the DICOM series and where to output this file to.

filepath = '/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT'

reader = sitk.ImageSeriesReader()
dcm_paths = reader.GetGDCMSeriesFileNames(filepath)
reader.SetFileNames(dcm_paths)
DICOM = sitk.ReadImage(dcm_paths)
DICOM_resampled = resample_DICOM(DICOM) #DICOM_resampled is an Image/Object not an array

print('================ NON-RESAMPLED DICOM DIMENSIONS ============')
print('Image size: ' + str(DICOM.GetSize()))
print('Image spacing: ' + str(DICOM.GetSpacing()))
print('Image direction: ' + str(DICOM.GetDirection()))
print('Image origin: ' + str(DICOM.GetOrigin()))

print('================ RESAMPLED DICOM DIMENSIONS ================')
print('Image size: ' + str(DICOM_resampled.GetSize()))
print('Image spacing: ' + str(DICOM_resampled.GetSpacing()))
print('Image direction: ' + str(DICOM_resampled.GetDirection()))
print('Image origin: ' + str(DICOM_resampled.GetOrigin()))
#===============================================================================================

#=========================== RESAMPLING THE MASK ===============================================

# Telling the code where to get the DICOMs and RTSTRUCT from
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
mask_3d = mask_3d_Lung_Right + mask_3d_Lung_Left

#Converting this array from boolean to float so that it can be converted to .nii file
mask_3d = mask_3d.astype(np.float32)

mask_3d_image = sitk.GetImageFromArray(mask_3d)



print('================ NON-RESAMPLED RTSTRUCT DIMENSIONS ==========')
print('Image size: ' + str(mask_3d_image.GetSize()))
print('Image spacing: ' + str(mask_3d_image.GetSpacing()))
print('Image direction: ' + str(mask_3d_image.GetDirection()))
print('Image origin: ' + str(mask_3d_image.GetOrigin()))

mask_3d_image = permute_axes(mask_3d_image, [1,2,0])
mask_3d_image.SetSpacing([DICOM.GetSpacing()[0], DICOM.GetSpacing()[1], DICOM.GetSpacing()[2]])
mask_3d_image.SetDirection(DICOM.GetDirection())
mask_3d_image.SetOrigin(DICOM.GetOrigin())
#mask_3d_image.SetSpacing([DICOM.GetSpacing()[2], DICOM.GetSpacing()[1], DICOM.GetSpacing()[0]])

# print(mask_3d_image.GetDirection())
# print(mask_3d_image.GetOrigin())

#mask_3d_image = permute_axes(mask_3d_image)

# print(mask_3d_image.GetDirection())
# print(mask_3d_image.GetOrigin())

mask_3d_image_resampled = resample_MASK(mask_3d_image)

print('================ RESAMPLED RTSTRUCT DIMENSIONS ==============')
print('Image size: ' + str(mask_3d_image_resampled.GetSize()))
print('Image spacing: ' + str(mask_3d_image_resampled.GetSpacing()))
print('Image direction: ' + str(mask_3d_image_resampled.GetDirection()))
print('Image origin: ' + str(mask_3d_image_resampled.GetOrigin()))

# print('=============================================================')
# print('mask array shape: ' + str(mask_3d.shape))
# print('mask image shape: ' +str(mask_3d_image.GetSize()))

#================================================================================================

#=========================== PERFORMING RESAMPLING ===================

#redefining mask_3d_image_resampled to mask_3d (array)
mask_3d = sitk.GetArrayFromImage(mask_3d_image_resampled)
X = sitk.GetImageFromArray(mask_3d)
X = sitk.GetArrayFromImage(X)

reader = sitk.ImageSeriesReader() #can hash out to just see mask
dcm_paths = reader.GetGDCMSeriesFileNames('/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT') #can hash out to just see mask
reader.SetFileNames(dcm_paths) #can hash out to just see mask
volume = reader.Execute() #can hash out to just see mask
volume = resample_DICOM(volume)

#============================================================================================

#=========================== WRITING THE RESAMPLED MASK AND DICOM ARRAYS AS A NIFTI FIL===================
mask_3d_resampled = sitk.GetArrayFromImage(mask_3d_image_resampled)
mask_3d_resampled = mask_3d_resampled.astype(np.float32)
mask_3d_image_resampled = sitk.GetImageFromArray(mask_3d_resampled)
sitk.WriteImage(mask_3d_image_resampled, "/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/resampled/LUNG1-001-MASK-resampled32bit.nii")
sitk.WriteImage(DICOM_resampled, "/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/resampled/LUNG1-001-DICOM-resampled.nii")
#============================================================================================

#=========================== COUNTING '1's ==================================================
# mask_3d_resampled = sitk.GetArrayFromImage(mask_3d_image_resampled)
# print(mask_3d_resampled.shape)

# numbers = np.arange(mask_3d_resampled.shape[2])
# slice_numbers = numbers + 1
# print(slice_numbers)

# true_counter=0
# for i in (slice_numbers - 1):
#   true_counter_i = 0
#   mask_test = mask_3d_resampled[:,:,i]
#   for row in mask_test:
#     for cell in row:
#         #cell = str(cell)
#         if cell == 1 :
#           true_counter_i +=1
#           #print(true_counter_i)
#           #print("True")
#   true_counter += true_counter_i
#   print("The number of True in slice " + str(i+1) + " is " + str(true_counter_i))
# print(true_counter)

# print(mask_3d_resampled[:,:,23])

#=============================================================================================