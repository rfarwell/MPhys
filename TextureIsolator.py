"""
This code makes a textured mask from the cropped CT and Mask .nii files.
It outputs the textured array as .nii
"""

import SimpleITK as sitk
import os
import numpy as np

main_folder_filepath = "/Volumes/Seagate_HDD/NSCLC_resampled_cropped_GTV-1/"

CT_filenames = []
mask_filenames = []

for filename in os.listdir(main_folder_filepath) :
    if "-CT" in filename :
        CT_filenames.append(filename)
    elif "RTSTRUCT" in filename :
        mask_filenames.append(filename)

print(CT_filenames)
print(mask_filenames)
