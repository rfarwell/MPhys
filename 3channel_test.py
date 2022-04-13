from curses import window
import skimage.io as io
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

# Read
img = io.imread('test_image.png')

read_filepath = "/Volumes/Seagate_HDD/Dilated_NSCLC_textured_masks_GTV-1/LUNG1-001-GTV-1.nii"
image = sitk.ReadImage(read_filepath)
array = sitk.GetArrayFromImage(image)


def window_and_level(image, level = -50, window = 600, original = False) :
  if original == True :
      min_val_array = np.amin(image)
      max_val_array = np.amax(image)
      print(f'Minimum pixel value in this original image (before window and level) = {min_val_array}')
      print(f'Maximum pixel value in this original image (before window and level) = {max_val_array}')
      range = max_val_array - min_val_array
      image -= min_val_array
      image /= range
      print(f'Minimum pixel value in this original image (after window and level) = {np.amin(image)}')
      print(f'Maximum pixel value in this original image (after window and level) = {np.amin(image)}')
      return image
  
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld

def isolate_micro_spread(image, level = -500, window = 800):
  micro_spread = window_and_level(image, level, window)
  for i in range(180):
    for j in range(180):
      for k in range(180):
        if micro_spread[i][j][k] == 1:
          micro_spread[i][j][k] = 0
  return micro_spread 

rgb_array = np.zeros((3,180,180), np.float32)
rgb_array[0,...]= window_and_level(array)[90]
rgb_array[1,...] = isolate_micro_spread(array)[90]
rgb_array[2,...] = window_and_level(array, original=True)[90]

# Split
red = rgb_array[0,:, :]
green = rgb_array[1,:, :]
blue = rgb_array[2,:, :]

# red = img[:, :, 0]
# green = img[:, :, 1]
# blue = img[:, :, 2]

# Plot
fig, axs = plt.subplots(2,2)

cax_00 = axs[0,0].imshow(img)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

cax_01 = axs[0,1].imshow(red, cmap='Reds')
fig.colorbar(cax_01, ax=axs[0,1])
axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

cax_10 = axs[1,0].imshow(green, cmap='Greens')
fig.colorbar(cax_10, ax=axs[1,0])
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

cax_11 = axs[1,1].imshow(blue, cmap='Blues')
fig.colorbar(cax_11, ax=axs[1,1])
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()

# Plot histograms
fig, axs = plt.subplots(3, sharex=True, sharey=True)

axs[0].hist(red.ravel(), bins=10)
axs[0].set_title('Red')
axs[1].hist(green.ravel(), bins=10)
axs[1].set_title('Green')
axs[2].hist(blue.ravel(), bins=10)
axs[2].set_title('Blue')

plt.show()