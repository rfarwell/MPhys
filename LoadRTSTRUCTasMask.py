import rt_utils
import sys
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt

# Load existing RT Struct. Requires the series path and existing RT Struct path
rtstruct = RTStructBuilder.create_from(
  dicom_series_path="LUNG1-001/CT", 
  rt_struct_path="LUNG1-001/RTSTRUCT/3-2.dcm"
)

# View all of the ROI names from within the image
print(rtstruct.get_roi_names())

# Loading the 3D Mask from within the RT Struct
mask_3d = rtstruct.get_roi_mask_by_name("GTV-1") #Put in the name of the ROI

# Display one slice of the region
first_mask_slice = mask_3d[:, :, 80] #the number defines which slice will be displayed
print(mask_3d.shape[2]) #prints the number of slices in the image (will be used when making an interactive plot)
# plt.imshow(first_mask_slice)
# plt.show()

# for i in range(mask_3d.shape[2]) :
#   plt.imshow(mask_3d[:,:,i])
#   plt.show()

with np.printoptions(threshold=sys.maxsize):
     print(mask_3d)




# with np.printoptions(threshold=sys.maxsize) :
#   print(mask_3d[:,:,80].shape)

# true_counter = 0
# mask_test = mask_3d[:,:,80]
# for row in mask_test:
#     for cell in row:
#         cell = str(cell)
#         if cell == "True" :
#           true_counter +=1
#           print(true_counter)
#           print("True")
# print(true_counter)