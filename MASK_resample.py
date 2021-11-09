"""
This code will use the ResampleImageFilter function to resample both the DICOM and Mask for a relevant DICOM series
"""

#========================== IMPORTING LIBRARIES =================================================
import rt_utils
import sys
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy
from scipy import stats
import os
reader = sitk.ImageSeriesReader()
#================================================================================================

#========================== RESAMPLING THE DICOM FILE ===========================================
Output_Spacing = [0.9765625, 0.9765625, 3.0]


def resample_DICOM(volume, interpolator = sitk.sitkLinear) :
    '''
    This function resample a volume to size 512 x 512 x 256 with spacing 1 x 1 x 4 (Good for our dataset)
    '''
    new_size = [512, 512, 134]
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([Output_Spacing[0], Output_Spacing[1], Output_Spacing[2]])
    resample.SetDefaultPixelValue(-1024)

    return resample.Execute(volume)


# Directing the code for where to look for the DICOM series and where to output this file to.

filepath = '/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT'

reader = sitk.ImageSeriesReader()
dcm_paths = reader.GetGDCMSeriesFileNames(filepath)
reader.SetFileNames(dcm_paths)
DICOM = sitk.ReadImage(dcm_paths)
DICOM_resampled = resample_DICOM(DICOM)

print('================ NON-RESAMPLED DICOM DIMENSIONS ============')
print('Image size: ' + str(DICOM.GetSize()))
print('Image spacing: ' + str(DICOM.GetSpacing()))


print('================ RESAMPLED DICOM DIMENSIONS ================')
print('Image size: ' + str(DICOM_resampled.GetSize()))
print('Image spacing: ' + str(DICOM_resampled.GetSpacing()))

#===============================================================================================

#=========================== RESAMPLING THE MASK ===============================================

rtstruct = RTStructBuilder.create_from(
  dicom_series_path="/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT", 
  rt_struct_path="/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT/3-2.dcm"
)

# Getting arrays for all the masks for the determined ROIs
mask_3d_Lung_Right = rtstruct.get_roi_mask_by_name("Lung-Right") 
mask_3d_Lung_Left = rtstruct.get_roi_mask_by_name("Lung-Left")
mask_3d_GTV_1 = rtstruct.get_roi_mask_by_name("GTV-1")
mask_3d_spinal_cord = rtstruct.get_roi_mask_by_name("Spinal-Cord")

# Setting what the desired mask is (for the case of a tumour we out GTV-1)
mask_3d = mask_3d_Lung_Right + mask_3d_Lung_Left

#Converting this array from boolean to binary
mask_3d = mask_3d + 1
mask_3d = mask_3d - 1

# Rotating the image to try and fix the axis labels
# mask_3d = np.rot90(mask_3d, 1, axes = (0,2))
# print('rotated mask array shape: ' + str(mask_3d.shape))

mask_3d_image = sitk.GetImageFromArray(mask_3d)
mask_3d_image.SetSpacing([DICOM.GetSpacing()[2], DICOM.GetSpacing()[1], DICOM.GetSpacing()[0]])

def permute_axes(volume) :
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder([2,1,0])

    return permute.Execute(volume)

mask_3d_image = permute_axes(mask_3d_image)

print('================ NON-RESAMPLED RTSTRUCT DIMENSIONS ==========')
print('Image size: ' + str(mask_3d_image.GetSize()))
print('Image spacing: ' + str(mask_3d_image.GetSpacing()))

def resample_MASK(volume, interpolator = sitk.sitkNearestNeighbor) :
    '''
    This function resample a volume to size 512 x 512 x 256 with spacing 1 x 1 x 4 (Good for our dataset)
    '''
    new_size = [134, 512, 512]
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing([Output_Spacing[2], Output_Spacing[1], Output_Spacing[0]])
    resample.SetDefaultPixelValue(0)

    return resample.Execute(volume)



mask_3d_image_resampled = resample_MASK(mask_3d_image)

print('================ RESAMPLED RTSTRUCT DIMENSIONS ==============')
print('Image size: ' + str(mask_3d_image_resampled.GetSize()))
print('Image spacing: ' + str(mask_3d_image_resampled.GetSpacing()))

print('=============================================================')
print('mask array shape: ' + str(mask_3d.shape))
print('mask image shape: ' +str(mask_3d_image.GetSize()))

#================================================================================================

#=========================== PLOTTING RESAMPLED IMAGES ONTO A SCROLLABLE PLOT ===================



class IndexTracker:
    def __init__(self, ax, X, volume):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.volume = sitk.GetArrayFromImage(volume) #can hash out to just see mask
        #print(self.volume.shape) #can hash out to just see mask
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.vol = ax.imshow(self.volume[self.ind], cmap = 'gray') #can hash out to just see mask
        self.im = ax.imshow(self.X[:, :, self.ind], alpha = 0.5)
        
        self.update()

        
    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.vol.set_data(self.volume[self.ind]) #can hash out to just see mask
        self.im.set_data(self.X[:, :, self.ind])
        
        self.ax.set_ylabel('GTV-1 of slice %s' % (self.ind + 1))
        self.im.axes.figure.canvas.draw()

fig, ax = plt.subplots(1, 1)

#redefining mask_3d_image_resampled to mask_3d (array)
mask_3d = sitk.GetArrayFromImage(mask_3d_image_resampled)
X = sitk.GetImageFromArray(mask_3d)
print(X.GetSize())
X = sitk.GetArrayFromImage(X)

reader = sitk.ImageSeriesReader() #can hash out to just see mask
dcm_paths = reader.GetGDCMSeriesFileNames('/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/-CT') #can hash out to just see mask
reader.SetFileNames(dcm_paths) #can hash out to just see mask
volume = reader.Execute() #can hash out to just see mask
volume = resample_DICOM(volume)
print(volume.GetSize())
tracker = IndexTracker(ax, mask_3d, volume)

print(volume.GetSize())
volume_array = sitk.GetArrayFromImage(volume)
print(volume_array.shape)

fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
#============================================================================================

#=========================== COUNTING '1's ==================================================
# mask_3d_resampled = sitk.GetArrayFromImage(mask_3d_image_resampled)
# print(mask_3d_resampled.shape)

# numbers = np.arange(mask_3d_resampled.shape[2])
# slice_numbers = numbers + 1
# print(slice_numbers)

# true_counter=0
# for i in (slice_numbers - 1):
#   true_counter_i = 0
#   mask_test = mask_3d_resampled[:,:,i]
#   for row in mask_test:
#     for cell in row:
#         #cell = str(cell)
#         if cell == 1 :
#           true_counter_i +=1
#           #print(true_counter_i)
#           #print("True")
#   true_counter += true_counter_i
#   print("The number of True in slice " + str(i+1) + " is " + str(true_counter_i))
# print(true_counter)

# print(mask_3d_resampled[:,:,23])


#=============================================================================================
print(DICOM.GetSpacing())