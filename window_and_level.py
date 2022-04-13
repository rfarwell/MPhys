import SimpleITK as sitk
import numpy as np
import sys, os

def window_and_level(image, level = -50, window = 350) :
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld

def window_and_level_original(image):
  min_val_array = np.amin(image)
  print(min_val_array)
  max_val_array = np.amax(image)
  print(max_val_array)
  range = max_val_array - min_val_array
  image -= min_val_array
  image /= range
  print(np.amin(image))
  print(np.amax(image))
  return image

def isolate_micro_spread(image, level = -500, window = 800):
  maxval = level + window/2
  minval = level - window/2
  micro_spread = np.clip(image, minval, maxval)
  micro_spread -=minval
  micro_spread *= 1/window
  for i in range(180):
    for j in range(180):
      for k in range(180):
        if micro_spread[i][j][k] == 1:
          micro_spread[i][j][k] = 0
  return micro_spread

read_filepath = "/Volumes/Seagate_HDD/Dilated_NSCLC_textured_masks_GTV-1"
write_filepath = "/Volumes/Seagate_HDD/WandLTesting"

counter = 0

for filename in os.listdir(read_filepath):
    counter+=1
    if counter > 5:
        sys.exit(1)
    mask_image = sitk.ReadImage(os.path.join(read_filepath, filename))
    mask_array = sitk.GetArrayFromImage(mask_image)
    # mask_array = isolate_micro_spread(mask_array)
    # mask_array = window_and_level(mask_array, window = 800, level = -500)
    mask_array = window_and_level_original(mask_array)
    mask_image = sitk.GetImageFromArray(mask_array)
    sitk.WriteImage(mask_image,os.path.join(write_filepath, filename))
    print(f"Finished window and levelling for {filename}")