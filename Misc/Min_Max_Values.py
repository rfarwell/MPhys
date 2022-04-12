import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

read_filepath = "/Volumes/Seagate_HDD/Dilated_NSCLC_textured_masks_GTV-1"

# mask_image = sitk.ReadImage(read_filepath)
# mask_array = sitk.GetArrayFromImage(mask_image)
# max_value = np.max(mask_array)
# min_value = np.min(mask_array)

filenames = []
for filename in os.listdir(read_filepath) :
    filenames.append(filename)
max_values = []
centre_pixel_values = []
counter=0
for i in range(len(filenames)):
    counter+=1
    mask_image = sitk.ReadImage(os.path.join(read_filepath, filenames[i]))
    mask_array = sitk.GetArrayFromImage(mask_image)
    # max_value = np.max(mask_array)
    # min_value = np.min(mask_array)
    # max_values.append(max_value)
    if mask_array[90][90][90] != -1024:
        print("=============================")
        print(filenames[i])
        print(mask_array[90][90][90])
        centre_pixel_values.append(mask_array[90][90][90]) # getting the value of the middle pixel
    # else:
    #     print(f"Centre pixel of {filenames[i]} is -1024")
    #     sys.exit(1)
    # print(f"{filenames[i]} --> Min = {min_value}, Max = {max_value}")


# print(np.max(max_values))
# print(np.min(max_values))

# plt.hist(max_values, bins = range(0,3500,100))
plt.hist(centre_pixel_values, bins = int((np.max(centre_pixel_values)-np.min(centre_pixel_values))/50)+1)
plt.title(f"Bin size = 50")
plt.xlabel("Centre pixel (tumour) value in textured mask")
plt.ylabel("Frequency")
plt.show()

