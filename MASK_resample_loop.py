"""
This code will use the ResampleImageFilter function to resample both the DICOM and Mask for a 
relevant DICOM series. Also outputs both the DICOM and masks as .nii files which can be used 
in 'worldmatch' to check that the mask and CT line up.

Rory Farwell : Last Edited (11/11/2021) (dd/mm/yyyy)
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

def resample_DICOM(volume, interpolator = sitk.sitkLinear, default_pixel_value = -1024) :
    '''
    This function resample a volume to size 512 x 512 x 512 with spacing defined from Output_Spacing
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(DICOM.GetDirection())
    resample.SetOutputOrigin(DICOM.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(default_pixel_value)

    return resample.Execute(volume)


def resample_MASK(volume, interpolator = sitk.sitkNearestNeighbor, default_pixel_value = 0) :
    '''
    This function resample a volume to size 512 x 512 x 256 with spacing 1 x 1 x 4 (Good for our dataset)
    '''
    print(volume.GetSize())
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(DICOM.GetDirection())
    resample.SetOutputOrigin(DICOM.GetOrigin())
    resample.SetSize(new_size)
    #resample.SetOutputSpacing([Output_Spacing[2], Output_Spacing[1], Output_Spacing[0]])
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(default_pixel_value)

    return resample.Execute(volume)


def permute_axes(volume, permutation_order) :
    """
    This function permutes the axes of the input volume.
    """
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder(permutation_order)

    return permute.Execute(volume)

#================================================================================================

#======================== LOOPING THROUGH ALL EXTERNALLY STORED CT AND RTSTRUCT FILES =======
number_of_iterations = 422
filenumbers = np.arange(number_of_iterations)
filenumbers = filenumbers + 1

filepath = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted'

for i in filenumbers :
    CT_read_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-CTUnknownStudyID'
    RTSTRUCT_initial_read_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCTUnknownStudyID'
    files_in_RTSTRUCT_folder = os.listdir('/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCTUnknownStudyID')
    RTSTRUCT_read_filename = str(files_in_RTSTRUCT_folder[0])
    RTSTRUCT_read_path = RTSTRUCT_initial_read_path + '/' + RTSTRUCT_read_filename
    
    CT_write_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_CT_and_RTSTRUCT/LUNG1-' + str('{0:03}'.format(i)) + '-CT.nii'
    RTSTRUCT_write_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_CT_and_RTSTRUCT/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCT.nii'
    
    reader = sitk.ImageSeriesReader()
    dcm_paths = reader.GetGDCMSeriesFileNames(CT_read_path)
    reader.SetFileNames(dcm_paths)
    DICOM = sitk.ReadImage(dcm_paths)
    DICOM_resampled = resample_DICOM(DICOM) #DICOM_resampled is an Image/Object not an array
    sitk.WriteImage(DICOM_resampled, CT_write_path)

    rtstruct = RTStructBuilder.create_from(
    dicom_series_path= CT_read_path, 
    rt_struct_path= RTSTRUCT_read_path
    )

    print(str(i) + ": " + str(rtstruct.get_roi_names())) # View all of the ROI names from within the image

    mask_3d_Lung_Right = rtstruct.get_roi_mask_by_name("Lung-Right") 
    mask_3d_Lung_Left = rtstruct.get_roi_mask_by_name("Lung-Left")
    mask_3d_GTV_1 = rtstruct.get_roi_mask_by_name("GTV-1")
    mask_3d_spinal_cord = rtstruct.get_roi_mask_by_name("Spinal-Cord")

    mask_3d = mask_3d_Lung_Right + mask_3d_Lung_Left

    mask_3d = mask_3d.astype(np.float32)

    mask_3d_image = sitk.GetImageFromArray(mask_3d)

    mask_3d_image = permute_axes(mask_3d_image, [1,2,0])
    mask_3d_image.SetSpacing(DICOM.GetSpacing())
    mask_3d_image.SetDirection(DICOM.GetDirection())
    mask_3d_image.SetOrigin(DICOM.GetOrigin())

    mask_3d_image_resampled = resample_MASK(mask_3d_image, sitk.sitkNearestNeighbor, 0)
    mask_3d_resampled = sitk.GetArrayFromImage(mask_3d_image_resampled)
    mask_3d_resampled = mask_3d_resampled.astype(np.float32)
    mask_3d_image_resampled = sitk.GetImageFromArray(mask_3d_resampled)
    mask_3d_image_resampled.SetDirection(DICOM.GetDirection())
    mask_3d_image_resampled.SetOrigin(DICOM.GetOrigin())
    # print(mask_3d_image_resampled.GetOrigin())
    sitk.WriteImage(mask_3d_image_resampled, RTSTRUCT_write_path)

    print('Completeted writing .nii files for CT and RTSTRUCT for file ' + str(i) + '.')


