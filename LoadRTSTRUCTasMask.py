import rt_utils
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
plt.imshow(first_mask_slice)
plt.show()