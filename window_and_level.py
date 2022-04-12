import SimpleITK as sitk
import numpy as np
import sys, os

def window_and_level(image, level = -50, window = 600) :
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld

read_filepath = "/Volumes/Seagate_HDD/Dilated_NSCLC_textured_masks_GTV-1"
write_filepath = "/Volumes/Seagate_HDD/WandLTesting"

counter = 0

for filename in os.listdir(read_filepath):
    counter+=1
    if counter > 5:
        sys.exit(1)
    mask_image = sitk.ReadImage(os.path.join(read_filepath, filename))
    mask_array = sitk.GetArrayFromImage(mask_image)
    mask_array = window_and_level(mask_array)
    mask_image = sitk.GetImageFromArray(mask_array)
    sitk.WriteImage(mask_image,os.path.join(write_filepath, filename))