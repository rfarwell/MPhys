"""
This code resamples the DICOM lung data (NSCLC-radiomics).
It does this using SimpleITK.
"""

#Importing relevant functions
from __future__ import print_function
import SimpleITK as sitk
import numpy as np #not sure whether this is needed
import scipy
from scipy import stats #not sure whether this is needed
import os
reader = sitk.ImageSeriesReader()

mode_x = 0.9765625
mode_y = 0.9765625
mode_z =3

number_of_iterations = 422+1
counter = 0

def resample_volume(volume, interpolator = sitk.sitkLinear, new_spacing = [mode_x, mode_y, mode_z]):
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
output_file = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_nifti_112/LUNG1-'
number_of_iterations = 2

filepath = '/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_Sorted'
number_of_patient = 0

for filename in os.listdir(filepath) :
    if "-CT" in filename :
        reader = sitk.ImageSeriesReader()
        dcm_paths = reader.GetGDCMSeriesFileNames(os.path.join(filepath, filename))

        reader.SetFileNames(dcm_paths)
        volume = sitk.ReadImage(dcm_paths)
        number_of_patient +=1
        _3_digit_number_of_patient = '{0:03}'.format(number_of_patient)
        print(_3_digit_number_of_patient)
        x = resample_volume(volume)
        number_of_patient += 1
        print(number_of_patient)
        #sitk.WriteImage(x, "/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_nifti_114/LUNG1-" + str(_3_digit_number_of_patient) + ".nii")
    else:
        pass

# for x in range(1, number_of_iterations) :
#    _3_digit_x = '{0:03}'.format(x) #formats the 'x' to the form 'yzx' e.g. 1 -> 001
#                                    # so that it fits the format of the naming scheme used
#                                    # e.g. LUNG1-001-CTUnknownStudyID
#    output_file_full = output_file + _3_digit_x +'.nii' # the '+.nii' ensures the output file is a NIfTI
#    directory_full = directory + str(_3_digit_x) + '-CTUnknownStudyID/'
#    dicom_paths = reader.GetGDCMSeriesFileNames(directory_full)
#    reader.SetFileNames(dicom_paths)
#    volume = sitk.ReadImage(dicom_paths)
#    resampled_volume = resample_volume(volume)
#    sitk.WriteImage(resample_volume, output_file_full) #writes the output file to output_file_full
#    counter += 1
#    print(counter) # Checking that the code is running and its progress

#    #The two lines below would open the file in ImageJ (Fiji)
#    """if ("SITK_NOSHOW" not in os.environ):
#            sitk.Show(image, "Dicom Series")"""



