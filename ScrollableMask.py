"""
This code will produce a viewer that allowd the user to scroll through all
of the slices of a particular DICOM series

In the future I will adapt this to allow input from the user, in the terminal.
The input would be: path to DICOM series, path to RTSTRUCT and ROI.

Rory Farwell 02/11/2021
"""
#=======================IMPORTING FUNCTIONS========================================
import rt_utils
import sys
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
#==================================================================================

#=======================PRE-REQUISITES=============================================

""" The below loads existing RT Struct. Requires the series path and existing RT Struct path """
rtstruct = RTStructBuilder.create_from(
  dicom_series_path="/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/CT", 
  rt_struct_path="/Users/roryfarwell/Documents/University/Year4/MPhys/DataOrg/LUNG1-001/RTSTRUCT/3-2.dcm"
)

""" The below prints all of the ROI names from the image metadata """
print(rtstruct.get_roi_names()) 


""" The below loads the 3D Mask, for the chosen ROI,
from the RT Struct file. """
mask_3d = rtstruct.get_roi_mask_by_name("GTV-1") 


""" The below produces an array with the length of the number of slices
in the DICOM series. This will be used when we label the axes of the plot. """
numbers = np.arange(mask_3d.shape[2])
slice_numbers = numbers + 1
#=================================================================================

#=======================MAIN BODY=================================================
"""
The code in this cell produces a matplotlib plot of the mask slices which
can be 'scrolled' through using the mouse scroll wheel. 
NOTE: this does not work in a Jupyter notebook and must be run in a pop up for
it to work. I need to see if I can fix this.
"""

class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('GTV-1 of slice %s' % (self.ind + 1))
        self.im.axes.figure.canvas.draw()

fig, ax = plt.subplots(1, 1)

X = mask_3d

tracker = IndexTracker(ax, mask_3d)

fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
#=======================END=======================================================
