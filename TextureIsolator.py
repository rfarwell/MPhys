"""
This code makes a textured mask from the cropped CT and Mask .nii files.
It outputs the textured array as .nii
"""

import SimpleITK as sitk
import os
import numpy as np

main_folder_filepath = "/Volumes/Seagate_HDD/NSCLC_resampled_cropped_GTV-1/"
write_folder_filepath = "/Volumes/Seagate_HDD/NSCLC_textured_masks_GTV-1/"

CT_filenames = []
mask_filenames = []

for filename in os.listdir(main_folder_filepath) :
    if "-CT" in filename :
        CT_filenames.append(filename)
    elif "-GTV-1" in filename :
        mask_filenames.append(filename)

# print(CT_filenames)
# print(mask_filenames)

for i in range(len(CT_filenames)) :
    temp_CT_filename = CT_filenames[i]
    temp_mask_filename = mask_filenames[i]
    temp_mask_image = sitk.ReadImage(os.path.join(main_folder_filepath, temp_mask_filename))
    temp_CT_image = sitk.ReadImage(os.path.join(main_folder_filepath, temp_CT_filename))

    print(f"Mask original size, origin, direction: {temp_mask_image.GetSpacing()}, {temp_mask_image.GetOrigin()}, {temp_mask_image.GetDirection()}.")
    print(f"CT original size, origin, direction: {temp_CT_image.GetSpacing()}, {temp_CT_image.GetOrigin()}, {temp_CT_image.GetDirection()}.")

    temp_mask_array = sitk.GetArrayFromImage(temp_mask_image)
    temp_CT_array = sitk.GetArrayFromImage(temp_CT_image)

    textured_mask_array = np.multiply(temp_mask_array, temp_CT_array + 1024) - 1024

    textured_mask_image = sitk.GetImageFromArray(textured_mask_array)
    textured_mask_image.SetSpacing(temp_mask_image.GetSpacing())
    textured_mask_image.SetOrigin(temp_mask_image.GetOrigin()) 
    textured_mask_image.SetDirection(temp_mask_image.GetDirection())

    print(f"{textured_mask_image.GetSpacing()}, {textured_mask_image.GetOrigin()},  {textured_mask_image.GetDirection()}")

    sitk.WriteImage(textured_mask_image, os.path.join(write_folder_filepath, temp_mask_filename))
    print(f"Finished writing {temp_mask_filename}")
