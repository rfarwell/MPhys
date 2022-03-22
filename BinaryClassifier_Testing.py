"""
This is the code that will load the best performing model from the training 
program

Rory Farwell and Patrick Hastings
"""

import os
print(f'Running {__file__}')

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch

from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import Module
from torch.nn import Conv3d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten, xlogy_
from torch import nn
from torch import reshape
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.io import read_image
from torch.optim import Adam
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary
import pickle

from scipy.ndimage import zoom, rotate

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
print(f'Using {device} device')


project_folder = "/home/rory_farwell1_gmail_com/data/rory_and_pat_data/"
clinical_data_filename = "382_metadata.csv"
print(os.path.join(project_folder, clinical_data_filename))
#====================================================================
#=================== CLASS DEFINITION ===============================
#====================================================================
class results :
    def __init__(self, expected, predicted) :
        self.expected = expected
        self.predicted = predicted

    def confusion_matrix(self):
        print(confusion_matrix(self.expected, self.predicted))
    
    def evaluate_results(self):
        self.true_positive_counter = 0
        self.true_negative_counter = 0
        self.false_positive_counter = 0
        self.false_negative_counter = 0
        for i in range(len(self.expected)) :
            if self.expected[i] == 1 and self.predicted[i] == 1 :
                self.true_positive_counter += 1
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> true positive')
            elif self.expected[i] == 0 and self.predicted[i] == 0 :
                self.true_negative_counter += 1
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> true negative')
            elif self.expected[i] == 0 and self.predicted[i] == 1 :
                self.false_positive_counter += 1 
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> false positive')
            elif self.expected[i] == 1 and self.predicted[i] == 0 :
                self.false_negative_counter += 1 
                # print(f'[{self.expected[i]},{self.predicted[i]}] -> false negative')
        return self.true_positive_counter, self.true_negative_counter, self.false_positive_counter, self.false_negative_counter

transform = transforms.Compose(
    [transforms.ToTensor()] #added 13/12/2021 to normalize the inputs. THIS NORMALIZES to mean = 0 and std = 1
)

class ImageDataset(Dataset) :
  def __init__(self, annotations, img_dir, transform = transform, target_transform = None, shift_augment = True, rotate_augment = True, scale_augment = True, flip_augment = True) :
    self.img_labels = annotations
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    self.shifts = shift_augment
    self.rotations = rotate_augment
    self.flips = flip_augment
    self.scales = scale_augment
    print(self.img_labels)

  def __len__(self) :
    return len(self.img_labels)

  def __getitem__(self,idx) :
    img_path = os.path.join(self.img_dir, self.img_labels[idx][0] + "-GTV-1.nii" )
    image_sitk = sitk.ReadImage(img_path)
    # ID = self.img_labels[idx][0]
    # print(f'ID: {ID}')
    image = sitk.GetArrayFromImage(image_sitk)
    label = self.img_labels[idx][1]

    # Augmentations
    if self.shifts:
      mx_x, mx_yz = 10, 10 
      # find shift values
      cc_shift, ap_shift, lr_shift = random.randint(-mx_x,mx_x), random.randint(-mx_yz,mx_yz), random.randint(-mx_yz,mx_yz)
      # pad for shifting into
      image = np.pad(image, pad_width=((mx_x,mx_x),(mx_yz,mx_yz),(mx_yz,mx_yz)), mode='constant', constant_values=-1024) # original is zero but appears to work better with -1024 (HU of air)
      # crop to complete shift
      image = image[mx_x+cc_shift:160+mx_x+cc_shift, mx_yz+ap_shift:160+mx_yz+ap_shift, mx_yz+lr_shift:160+mx_yz+lr_shift]

    if self.rotations and random.random() < 0.5 : # normal is 0.5
      roll_angle = np.clip(np.random.normal(loc=0,scale=3), -15, 15)
      # print(f'Rotation by angle {roll_angle} applied.')
      #print(roll_angle)
      image = self.rotation(image, roll_angle, rotation_plane=(1,2))

    if self.scales and random.random() < 0.5 : # normal is 0.5
      # same here -> zoom between 80-120%
      scale_factor = np.clip(np.random.normal(loc=1.0,scale=0.05), 0.7, 1.3)
      # print(f'Scaled by factor {scale_factor}.')
      image = self.scale(image, scale_factor)
    
    if self.flips and random.random() < 0.5 : # normal is 0.5
        # print(f'Left-right flip applied')
        image = np.flipud(image)
    
    image = window_and_level(image)

    if self.transform :
      image = self.transform(image)
    if self.target_transform :
      label = self.target_transform(label)
    return image,label
  
  def rotation(self, image, rotation_angle, rotation_plane):
      # rotate the image or mask using scipy rotate function
      order, cval = (3, -1024)
      return rotate(input=image, angle=rotation_angle, axes=rotation_plane, reshape=False, order=order, mode='constant', cval=cval)
    
  def scale(self, image, scale_factor):
      # scale the image or mask using scipy zoom function
      order, cval = (3, -1024)
      height, width, depth = image.shape
      zheight = int(np.round(scale_factor*height))
      zwidth = int(np.round(scale_factor*width))
      zdepth = int(np.round(scale_factor*depth))
      # zoomed out
      if scale_factor < 1.0:
          new_image = np.full_like(image, cval)
          ud_buffer = (height-zheight) // 2
          ap_buffer = (width-zwidth) // 2
          lr_buffer = (depth-zdepth) // 2
          new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth, lr_buffer:lr_buffer+zdepth] = zoom(input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
          return new_image
      elif scale_factor > 1.0:
          new_image = zoom(input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
          ud_extra = (new_image.shape[0] - height) // 2
          ap_extra = (new_image.shape[1] - width) // 2
          lr_extra = (new_image.shape[2] - depth) // 2
          new_image = new_image[ud_extra:ud_extra+height, ap_extra:ap_extra+width, lr_extra:lr_extra+depth]
          return new_image
      return image

class CNN(nn.Module):   
    def __init__(self):
      super(CNN, self).__init__()
      self.conv1 = nn.Conv3d(1,32,3,1)
      self.pool = nn.MaxPool3d(2,2)
      self.avg_pool = nn.AvgPool3d(8)
      self.conv2 = nn.Conv3d(32,64,3,1)
      self.conv3 = nn.Conv3d(64,128,3,1)
      self.conv4 = nn.Conv3d(128,256,3,1)
      self.conv5 = nn.Conv3d(256,64,1,1)
      self.conv6 = nn.Conv3d(64,16,1,1)
      self.conv7 = nn.Conv3d(16,2,1,1)
      self.softmax = nn.Softmax()

    # Defining the forward pass  (NIN method)  
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.avg_pool(self.conv7(x))
        x = self.softmax(x)
        x = x.view(-1,2)
        return x
        
# Load the network parameters from the training stages

import sys
if len(sys.argv) < 2:
    print("Correct usage is: python Binary_Classifier_Testing.py <file path of the saved network>")
    sys.exit(1)

model = CNN().to(device) # Send the CNN to the device
model.load_state_dict(torch.load(sys.argv[1]))

def convert_to_one_hot_labels(images, labels) :
    """
    This function converts the labels to one-hot labels so that they will work with the BCEwithLogitsLoss
    """
    hot_labels = torch.empty((images.shape[0], 2))
    
    for index in range(len(labels)) :
        if labels[index] == 0 :
            hot_labels[index,0] = 1
            hot_labels[index,1] = 0
        elif labels[index] == 1 :
            hot_labels[index, 0] = 0
            hot_labels[index, 1] = 1
    
    return hot_labels


def window_and_level(image, level = -600, window = 1500) :
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld

def testing_loop():
  print("---- Currently testing the network on unseen data ----")
  model.eval()

  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    counter = 0
    for images, labels in test_dataloader :
        # counter+=1
        # print(counter)
        images = images = reshape(images, (images.shape[0],1 ,160,160,160))
        images = images.float()
        hot_labels = convert_to_one_hot_labels(images, labels)

        images = images.to(device)
        hot_labels = hot_labels.to(device)
        outputs = model(images)
        # max returns (value, index) 
        _,predictions = torch.max(outputs, 1)
        _,targets = torch.max(hot_labels,1)
        #print(f'predictions: {predictions}')
        #print(f'targets: {targets}')
        n_samples += hot_labels.shape[0]
        n_correct += (predictions == targets).sum().item()
        #print(f'n_correct = {n_correct}. n_samples = {n_samples}')

        labels_numpy = labels.numpy()
        # print(f"labels_numpy = {labels_numpy}")

        for index in range(labels_numpy.size) :
            testing_targets.append(labels_numpy[index])
        
            # print(f"epoch_validation_targets = {epoch_validation_targets}")

        predictions_numpy = predictions.cpu().numpy()
        for index in range(predictions_numpy.size) :
            testing_predictions.append(predictions_numpy[index])

    
    acc = (100*n_correct)/n_samples

    return acc

open_file = open("testing_data_list.pkl", "rb")
outcomes_test = pickle.load(open_file)
open_file.close()
#print(outcomes_test)

test_data = ImageDataset(outcomes_test, os.path.join(project_folder, "textured_masks"), transform = transform, target_transform = None, shift_augment = False, rotate_augment = False, scale_augment = False, flip_augment = False) 
test_dataloader = DataLoader(test_data, batch_size = 4, shuffle = False)

testing_targets = []
testing_predictions = []

testing_accuracy = testing_loop()

testing_results = results(testing_targets, testing_predictions)
print(f"Targets: {testing_targets}")
print(f"Predictions: {testing_predictions}")
print(f'(TP, TN, FP, FN): {testing_results.evaluate_results()}')


