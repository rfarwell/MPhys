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

from MASK_resample_single import DICOM_resampled, DICOM_series_path
reader = sitk.ImageSeriesReader()
#================================================================================================

#========================== DEFINING RESAMPLING VARIABLES =======================================
Output_Spacing = [1, 1, 1]
new_size = [512, 512, 512]
#================================================================================================

#========================== DEFINING FUNCTIONS ==================================================
def resample_volume(volume, interpolator, default_pixel_value) :
    '''
    This function resample a volume to size 512 x 512 x 512 with spacing 1 x 1 x 1.

    It will be used in the resampling functions for both the DICOM series and the mask images.
    To ensure they overlap well when put into a program such as Worldmatch the direction and origin of both
    resampled volumes are set to the direction and origin of the original DICOM series.

    Rory Farwell and Patrick Hastings (14/11/2021)
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
    It will be used on the mask because SimpleITK seems to flip the axes
    at some stage in this process.

    Patrick Hastings and Rory Farwell (16/11/2021) 
    """
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder(permutation_order)

    return permute.Execute(volume)

def resample_DICOM(interpolator = sitk.sitkLinear, default_pixel_value = -1024) :
    """
    This function will do the whole resampling process on the DICOM series and will make use
    of the earlier defined resample_volume function.

    Rory Farwell and Patrick Hastings (14/11/2021)
    """
    global DICOM #Means that DICOM will be defined globally
    DICOM_paths = reader.GetGDCMSeriesFileNames(DICOM_series_path)
    reader.SetFileNames(DICOM_paths)
    DICOM = sitk.ReadImage(DICOM_paths)
    """DICOM_resampled = resample_volume(DICOM, interpolator, default_pixel_value) #DICOM_resampled is an Image/Object not an array

    return DICOM_resampled"""
    return DICOM

def resample_GTV_1_MASK(interpolator = sitk.sitkNearestNeighbor, default_pixel_value = 0, Regions_of_Interest = ["Lung-Right", "Lung-Left"]) :
    """
    This function will perform the whole resampling process on a mask produced from an RTSTRUCT
    file and will make use of the earlier defined resample_volume function.

    Rory Farwell and Patrick Hastings (14/11/2021)
    """
    rtstruct = RTStructBuilder.create_from(DICOM_series_path, RTSTRUCT_path) # Telling the code where to get the DICOMs and RTSTRUCT from
    
    # print(str(i) + ': ' + str(rtstruct.get_roi_names())) # View all of the ROI names from within the image
    
    #Getting arrays for all the masks for the determined ROIs
    #Note that these are only the ROIs for LUNG1-001. Other patients may have different ROIs which is something I need to check.
    mask_3d_Lung_Right = rtstruct.get_roi_mask_by_name("Lung-Right") 
    mask_3d_Lung_Left = rtstruct.get_roi_mask_by_name("Lung-Left")
    # mask_3d_GTV_1 = rtstruct.get_roi_mask_by_name("GTV-1")
    # mask_3d_spinal_cord = rtstruct.get_roi_mask_by_name("Spinal-Cord")

    mask_3d = mask_3d_Lung_Left + mask_3d_Lung_Right
    
    mask_3d = mask_3d.astype(np.float32) #Converting this array from boolean to float so that it can be converted to .nii file

    mask_3d_image = sitk.GetImageFromArray(mask_3d) #Converting mask_3d array to an image

    mask_3d_image = permute_axes(mask_3d_image, [1,2,0]) #permuting the axes of mask_2d because SimpleITK changes the axes ordering
    mask_3d_image.SetSpacing(DICOM.GetSpacing())
    mask_3d_image.SetDirection(DICOM.GetDirection())
    mask_3d_image.SetOrigin(DICOM.GetOrigin())

    mask_3d_image_resampled = resample_volume(mask_3d_image, interpolator, default_pixel_value)

    return mask_3d_image_resampled

def resample_ALL_GTV_MASK(interpolator = sitk.sitkNearestNeighbor, default_pixel_value = 0, Regions_of_Interest = ["Lung-Right", "Lung-Left"]) :
    """
    This function will perform the whole resampling process on a mask produced from an RTSTRUCT
    file and will make use of the earlier defined resample_volume function.

    Rory Farwell and Patrick Hastings (14/11/2021)
    """
    rtstruct = RTStructBuilder.create_from(DICOM_series_path, RTSTRUCT_path) # Telling the code where to get the DICOMs and RTSTRUCT from
    
    #Getting arrays for all the masks for the determined ROIs
    #Note that these are only the ROIs for LUNG1-001. Other patients may have different ROIs which is something I need to check.
    mask_3d = []

    ROIs = rtstruct.get_roi_names()

    for ROI in ROIs :
        mask_3d_temp = rtstruct.get_roi_mask_by_name(str(ROI))
        if "GTV" or "gtv" in ROI :
            mask_3d = mask_3d + mask_3d_temp

    
    mask_3d = mask_3d.astype(np.float32) #Converting this array from boolean to float so that it can be converted to .nii file

    mask_3d_image = sitk.GetImageFromArray(mask_3d) #Converting mask_3d array to an image

    mask_3d_image = permute_axes(mask_3d_image, [1,2,0]) #permuting the axes of mask_2d because SimpleITK changes the axes ordering
    mask_3d_image.SetSpacing(DICOM.GetSpacing())
    mask_3d_image.SetDirection(DICOM.GetDirection())
    mask_3d_image.SetOrigin(DICOM.GetOrigin())

    mask_3d_image_resampled = resample_volume(mask_3d_image, interpolator, default_pixel_value)

    return mask_3d_image_resampled

def get_filenumbers() :
    """
    This function runs through all the files and tests whether both the DICOM and RTSTRUCT files can be opened.
    If they can then they are removed from the 'filenumbers_tested' list which will be used for when creating
    .nii files of both the masks and the DICOM series.

    Patrick Hastings and Rory Farwell (16/11/2021)
    """
    number_of_iterations = 422
    filenumbers = np.arange(number_of_iterations)
    filenumbers = filenumbers + 1

    filenumbers_tested = filenumbers
    counter = 0

    for i in filenumbers :
        DICOM_series_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-CTUnknownStudyID'
        RTSTRUCT_initial_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCTUnknownStudyID'
        files_in_RTSTRUCT_folder = os.listdir('/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCTUnknownStudyID')
        RTSTRUCT_read_filename = str(files_in_RTSTRUCT_folder[0])
        RTSTRUCT_path = RTSTRUCT_initial_path + '/' + RTSTRUCT_read_filename
        counter += 1
        if counter%10 == 0 :
            print(counter)
    
            try :
                """
                Try to read both the RTSTRUCT and DICOM series.
                """
                rtstruct = RTStructBuilder.create_from(DICOM_series_path, RTSTRUCT_path)
                DICOM_paths = reader.GetGDCMSeriesFileNames(DICOM_series_path)
                reader.SetFileNames(DICOM_paths)
                DICOM = sitk.ReadImage(DICOM_paths)
        
                ROIs = rtstruct.get_roi_names()
                for ROI in ROIs :
                    str(ROI)
                    if "pre-op" in ROI :
                        filenumbers_tested.remove(i)
                        
            except :
                """
                If unable to read in both RTSTRUCT and DICOM series.
                """
                filenumbers_tested.remove(i)
                print("Unable to read either the DICOM series or the RTSTRUCT for LUNG1-" + str('{0:03}'.format(i)) + ".")

    print(filenumbers_tested)
    print(len(filenumbers_tested))
    
    return filenumbers_tested

#================================================================================================

#======================== LOOPING THROUGH ALL EXTERNALLY STORED CT AND RTSTRUCT FILES =======

filenumbers_tested = get_filenumbers()
print('=================================================================================')

for i in filenumbers_tested :
    DICOM_series_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-CTUnknownStudyID'
    RTSTRUCT_initial_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCTUnknownStudyID'
    files_in_RTSTRUCT_folder = os.listdir('/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted/LUNG1-' + str('{0:03}'.format(i)) + '-RTSTRUCTUnknownStudyID')
    RTSTRUCT_read_filename = str(files_in_RTSTRUCT_folder[0])
    RTSTRUCT_path = RTSTRUCT_initial_path + '/' + RTSTRUCT_read_filename
    
    DICOM_write_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_CT_and_RTSTRUCT/LUNG1-' + str('{0:03}'.format(i)) + '-CT.nii'
    GTV_1_MASK_write_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_CT_and_RTSTRUCT/LUNG1-' + str('{0:03}'.format(i)) + '-GTV-1.nii'
    ALL_GTV_MASK_write_path = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_CT_and_RTSTRUCT/LUNG1-' + str('{0:03}'.format(i)) + '-ALL_GTV.nii'


    GTV_1_mask_3d_image_resampled = resample_GTV_1_MASK()
    sitk.WriteImage(GTV_1_mask_3d_image_resampled, GTV_1_MASK_write_path)
    print('Completeted writing .nii file for GTV-1 MASK for LUNG1-' + str('{0:03}'.format(i)) + '.')

    ALL_GTV_mask_3d_image_resampled = resample_ALL_GTV_MASK()
    sitk.WriteImage(ALL_GTV_mask_3d_image_resampled, ALL_GTV_MASK_write_path)
    print('Completeted writing .nii file for ALL GTV MASK for LUNG1-' + str('{0:03}'.format(i)) + '.')
    

    DICOM = resample_DICOM()
    DICOM_resampled = resample_DICOM()
    sitk.WriteImage(DICOM_resampled, DICOM_write_path)
    print('Completeted writing .nii file for CT for LUNG1-' + str('{0:03}'.format(i)) + '.')
    print("===========================================================================")
