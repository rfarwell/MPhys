"""
This code will read in the .nii files produced by Quick_MASK_resample_loop.py or MASK_resample_loop.py
and crop the arrays.

The method taken, for each patient, will be:
    1. Determine the centre of mass for the tumour.
    2. Find the largest distance of the tumour from the centre of the mass.
    3. Crop the image to these dimensions 

Developed from Patrick's code at https://github.com/PHastings37/Mphys-proj/blob/main/image%20cropper.py
"""

from typing import ForwardRef
import numpy as np
import os
import SimpleITK as sitk
from colorama import Fore
from colorama import Style

nifty_path = "/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_CT_and_RTSTRUCT"
output_path = "/Volumes/Extreme_SSD/MPhys/TCIA_Data/NSCLC-Radiomics/NSCLC_resampled_cropped_GTV-1"

def get_CoM(input_mask) :
    """
    A function to find the Centre of Mass (CoM) of the tumout from its mask
    
    x_coord, y_coord and z_coord are the x,y,z coordinates of the CoM respecitvely
    """
    positions = np.argwhere(input_mask)
    x_coord = np.round(np.average(positions[:,0]))
    y_coord = np.round(np.average(positions[:,1]))
    z_coord = np.round(np.average(positions[:,2]))
    return (x_coord, y_coord, z_coord)

def largest_gtv_finder(input_mask, CoMs) :
    """
    A function that find the difference between the lowest index and highest index of x,y,z
    and returns the furthest distance above and below the CoM of the tumour in x,y,z
    """
    max_distances = []
    positions = np.argwhere(input_mask)
    CoM = CoMs.pop()
    max_distances.append(np.abs(np.max(positions[:, 0]) - CoM[0]))
    max_distances.append(np.abs(np.min(positions[:, 0]) - CoM[0]))
    max_distances.append(np.abs(np.max(positions[:, 1]) - CoM[1]))
    max_distances.append(np.abs(np.min(positions[:, 1]) - CoM[1]))
    max_distances.append(np.abs(np.max(positions[:, 2]) - CoM[2]))
    max_distances.append(np.abs(np.min(positions[:, 2]) - CoM[2]))
    print(f"max_distances array: {max_distances}")
    largest_distance = np.max(max_distances)
    CoMs.append(CoM)
    return largest_distance

def cropping(input_array, CoM_array, cropping_size, filename) :
    """
    Cropping function that has a mask or CT inputted and returns the cropped versions of these.
    """
    xstart = CoM_array[0] - cropping_size
    xend = CoM_array[0] + cropping_size
    ystart = CoM_array[1] - cropping_size
    yend = CoM_array[1] + cropping_size
    zstart = CoM_array[2] - cropping_size
    zend = CoM_array[2] + cropping_size
    xstart = xstart.astype(int)
    xend = xend.astype(int)
    ystart = ystart.astype(int)
    yend = yend.astype(int)
    zstart = zstart.astype(int)
    zend = zend.astype(int)

    coords = []
    coords.extend([xstart, xend, ystart, yend, zstart, zend])
    coords = [0 if i < 0 else i for i in coords]
    coords = [512 if i > 512 else i for i in coords]
    sub_zero = True
    cropped_array = input_array[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]
    print(f"Cropped_array shape: {cropped_array.shape}")
    if cropped_array.shape != (cropping_size * 2, cropping_size * 2, cropping_size * 2) and sub_zero == True :
        print(f"File {filename} is being padded")
        if xstart < 0 :
            x_pad_before = np.abs(xstart)
        else :
            x_pad_before = 0
        
        if ystart < 0 :
            y_pad_before = np.abs(ystart)
        else :
            y_pad_before = 0
        
        if zstart < 0 :
            z_pad_before = np.abs(zstart)
        else :
            z_pad_before = 0

        if xend > 511 :
            x_pad_after = 511 - xend
        else :
            x_pad_after = 0

        if yend > 511 :
            y_pad_after = 511 - yend
        else :
            y_pad_after = 0

        if zend > 511 :
            z_pad_after = 511 - zend
        else :
            z_pad_after = 0
        
        if "-CT" in filename :
            """
            Background for cropped image is -1024 not 0
            """
            cropped_array = np.pad(cropped_array, [(x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)], mode = 'constant', constant_values = [(-1024, -1024), (-1024, -1024), (-1024, -1024)])
        
        elif "GTV-1" in filename or "ALL" in filename :
            cropped_array = np.pad(cropped_array, [(x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)], mode = 'constant', constant_values = [(0, 0), (0, 0), (0, 0)])

        print(f"Cropped-array shape: {cropped_array.shape}")
    return cropped_array


CoMs = []
largest_tumour_axis = 0
temp_largest_tumour_axis = 0
print("========================PROGRAM STARTING========================")
#============================== CROPPING GTV-1 MASKS ===================
print(f"{Fore.YELLOW}Read path: {nifty_path}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}Write path: {output_path}{Style.RESET_ALL}")

for filename in os.listdir(nifty_path) :
    if "-GTV-1" in filename :
        print(filename)
    else:
        continue

for filename in os.listdir(nifty_path) :
    
    if"-GTV-1" in filename :
        """
        Loops over all files looking for masks of GTV-1. Finds CoM of the tumour from the mask array
        Finds the largest axis in that tummout
        Finds the largest overall distance from CoM to edge of a tumour which defines the size of
        our crop
        """
        print(f"Currently determining CoM and largest axis size for {filename}")
        GTV_1_mask_image = sitk.ReadImage(os.path.join(nifty_path,filename))
        GTV_1_mask_array = sitk.GetArrayFromImage(GTV_1_mask_image)
        CoM_temp = get_CoM(GTV_1_mask_array)
        CoMs.append(CoM_temp)
        print(f"After processing {filename} CoMs (array) length: {len(CoMs)}")
        temp_largest_tumour_axis = largest_gtv_finder(GTV_1_mask_array, CoMs)
        print(f"Largest tumour axis of {filename} : {temp_largest_tumour_axis}")
        if temp_largest_tumour_axis > largest_tumour_axis :
            largest_tumour_axis = temp_largest_tumour_axis
            print(f"{Fore.GREEN}Largest tumour axis updated to {largest_tumour_axis}{Style.RESET_ALL}")
    else :
        continue

cropping_size = largest_tumour_axis + 15 # +15 because we want to pad by 1.5cm in each direction
print(f"The cropping size is {cropping_size}")
print(f"The length of CoMs is: {len(CoMs)}")


""" Will need to repeat this step for ALL_GTV masks and CTs"""
counter = -0.5
for filename in os.listdir(nifty_path) :
    if "ALL_GTV" in filename :
        continue
    else :
        counter += 0.5 # avoiding the index issues previously experienced that was due to the removal of some data during the resampling process
        print(filename)
        index = np.floor(counter)
        index = int(index)
        print(index)
        CoM_index = CoMs[index]
        print(f"CoM: {CoM_index}")

        image = sitk.ReadImage(os.path.join(nifty_path, filename))
        print(f"Original image size: {image.GetSize()}")
        print(f"Original image origin: {image.GetOrigin()}")
        array = sitk.GetArrayFromImage(image)
        print(f"Original array size: {array.shape}")

        cropped_array = cropping(array, CoM_index, cropping_size, filename)

        print(f"Cropped array shape : {cropped_array.shape}")

        cropped_image = sitk.GetImageFromArray(cropped_array)
        cropped_image.SetDirection(image.GetDirection())
        cropped_image.SetSpacing(image.GetSpacing())
        cropped_image.SetOrigin(image.GetOrigin())
        sitk.WriteImage(cropped_image, f"{output_path}/{filename}.nii")
#=======================================================================  

