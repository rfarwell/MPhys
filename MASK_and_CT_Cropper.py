"""
This code will read in the .nii files produced by Quick_MASK_resample_loop.py or MASK_resample_loop.py
and crop the arrays.

The method taken, for each patient, will be:
    1. Determine the centre of mass for the tumour.
    2. Find the largest distance of the tumour from the centre of the mass.
    3. Crop the image to these dimensions 
"""
print(1258/3)