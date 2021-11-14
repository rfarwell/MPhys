"""
This code will allow the user to define paths to read a CT DICOM series and an RTSTRUCT file, for the same
patient, and to define the path to write the CT series and a binary mask as separate Nifti files.

During this process, both will be resampled to a user defined spacing and size and the mask will be resampled
using the direction and origin of the original DICOM series. This is so that when viewed in a program such as 
Worldmatc they will overlap eachother.

This code will for the basis for the MASK_resample_loop.py file which will loop this process over a large dataset.

Rory Farwell and Patrick Hastings (14/11/2021) (dd/mm/yyyy)
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

#========================== DEFINING PATHS ======================================================
DICOM_series_path = "/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT"
RTSTRUCT_path = "/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT/3-2.dcm"
#================================================================================================


#========================== DEFINING FUNCTIONS ==================================================
def resample_volume(volume, interpolator, default_pixel_value) :
    '''
    This function resample a volume to size 512 x 512 x 512 with spacing 1 x 1 x 1 (need to test different
    sizes and spacings for our dataset).

    This function will be used in the resampling functions for both the DICOM series and the mask images.
    To ensure they overlap well when put into a program such as Worldmatch the direction and origin of both
    resampled volumes are set to the direction and origin of the original DICOM series.

    Rory Farwell and Patrick Hastings (14/11/2021) (dd/mm/yyyy)
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(DICOM.GetDirection())
    resample.SetOutputOrigin(DICOM.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(default_pixel_value)

    return resample.Execute(volume)


def permute_axes(volume, permutation_order) :
    """
    This function permutes the axes of the input volume.
    It will be used on the 
    """
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder(permutation_order)

    return permute.Execute(volume)

def resample_DICOM(interpolator = sitk.sitkLinear, default_pixel_value = -1024) :
    """
    This function will do the whole resampling process on the DICOM series and will make use
    of the earlier defined resample_volume function.

    Rory Farwell and Patrick Hastings (14/11/2021) (dd/mm/yyyy)
    """
    global DICOM #Means that DICOM will be defined globally

    reader = sitk.ImageSeriesReader()
    DICOM_paths = reader.GetGDCMSeriesFileNames(DICOM_series_path)
    reader.SetFileNames(DICOM_paths)
    DICOM = sitk.ReadImage(DICOM_paths)
    DICOM_resampled = resample_volume(DICOM, interpolator, default_pixel_value) #DICOM_resampled is an Image/Object not an array

    return DICOM_resampled

def resample_MASK(interpolator = sitk.sitkNearestNeighbor, default_pixel_value = 0, Regions_of_Interest = ["Lung-Right", "Lung-Left"]) :
    """
    This function will perform the whole resampling process on a mask produced from an RTSTRUCT
    file and will make use of the earlier defined resample_volume function.

    Rory Farwell and Patrick Hastings (14/11/2021) (dd/mm/yyyy)
    """

    rtstruct = RTStructBuilder.create_from(DICOM_series_path, RTSTRUCT_path) # Telling the code where to get the DICOMs and RTSTRUCT from

    # Getting arrays for all the masks for the determined ROIs
    mask_3d_Lung_Right = rtstruct.get_roi_mask_by_name("Lung-Right") 
    mask_3d_Lung_Left = rtstruct.get_roi_mask_by_name("Lung-Left")
    mask_3d_GTV_1 = rtstruct.get_roi_mask_by_name("GTV-1")
    mask_3d_spinal_cord = rtstruct.get_roi_mask_by_name("Spinal-Cord")

    mask_3d = mask_3d_Lung_Left + mask_3d_Lung_Right
    
    mask_3d = mask_3d.astype(np.float32) #Converting this array from boolean to float so that it can be converted to .nii file

    mask_3d_image = sitk.GetImageFromArray(mask_3d) #Converting mask_3d array to an image

    mask_3d_image = permute_axes(mask_3d_image, [1,2,0]) #permuting the axes of mask_2d because SimpleITK changes the axes ordering
    mask_3d_image.SetSpacing(DICOM.GetSpacing())
    mask_3d_image.SetDirection(DICOM.GetDirection())
    mask_3d_image.SetOrigin(DICOM.GetOrigin())

    mask_3d_image_resampled = resample_volume(mask_3d_image, interpolator, default_pixel_value)

    return mask_3d_image_resampled

#================================================================================================

#========================== FINAL CODE ==========================================================
mask_3d_image_resampled = resample_MASK()
DICOM_resampled = resample_DICOM()
sitk.WriteImage(mask_3d_image_resampled, "/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/resampled/LUNG1-001-MASK-resampled32bit.nii")
sitk.WriteImage(DICOM_resampled, "/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/resampled/LUNG1-001-DICOM-resampled.nii")
print("Finished writing files.")
#================================================================================================



