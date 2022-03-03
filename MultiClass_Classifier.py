"""
This code trains, validates and tests a multi-class classifying CNN

The inputs to the network are 264 x 264 x 264 textured masks of NSCLC pre-treatment CT scans.

The network makes predictions of the range of time in which a patient will die.

Rory Farwell and Patrick Hastings 08/02/2022

"""
#===================================================================================================================
#======================= IMPORTING FUNCTIONS =======================================================================
#===================================================================================================================
import os
print(f'Running {__file__}')
# Un hash below if on Google Colab
# !pip install torch 
# !pip install torchvision
# !pip install opencv-contrib-python
# !pip install scikit-learn
# !pip install SimpleITK

import numpy as np
import random
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import math

from mpl_toolkits.mplot3d import Axes3D
from torch.nn import Module
from torch.nn import Conv3d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
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

from scipy.ndimage import zoom, rotate

#=================== USER INPUTS ===================================================================================
import sys # Will be used to allow the user to input the desired save name of 
if len(sys.argv) < 6 :
  print("Error: User inputs are wrong.")
  print("Correct usage: '/content/gdrive/MyDrive/University/Year_4_Sem_2/MultiClass_Classifier.py' <Number of epochs> <Time increment> <Upper time limit, in years> <Plot save name.png>")
  sys.exit(1)

#NUMBER OF EPOCHS
num_epochs = int(sys.argv[1])

#TIME INCREMENT
user_choice_of_time_increment = float(sys.argv[2])

#MAXIMUM TIME IN YEARS
maximum_time_in_years = int(sys.argv[3])
maximum_time_in_days = maximum_time_in_years * 365

number_of_categories = math.ceil(maximum_time_in_days/user_choice_of_time_increment) + 1

#WHETHER TO USE FULL DATASET OR PARTIAL DATASET
if sys.argv[4].lower() == 'true':
  print('You chose to run the full dataset.')
  full_dataset_choice = True
elif sys.argv[4].lower() == 'false':
  print('You chose to run the partial dataset.')
  full_dataset_choice = False
else:
  print('Error: Your input for whether to use the full dataset must be "True" or "False".')
  sys.exit(1)

plot_filename = sys.argv[5]
print(plot_filename)
plot_folder_path = "/content/gdrive/MyDrive/University/Year_4_Sem_2/MPhys_Plots/"


#===================================================================================================================
#=================== COLAB SPECIFIC CODE ===========================================================================
#===================================================================================================================
# Connecting to the GoogleDrive and mounting to the specific folder
# from google.colab import drive
# drive.mount('/content/gdrive')

#===================================================================================================================
#=================== SELECT DEVICE =================================================================================
#===================================================================================================================

# Connect to GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
# /content/gdrive/MyDrive/MPhys/Data/COLAB-Clinical-Data.csv
# Specify project folder location

"""
For Rory (short patient list): /content/gdrive/MyDrive/MPhys/Data/COLAB-Clinical-Data.csv
For Rory (full patient list): /content/gdrive/MyDrive/Data/NSCLC-Radiomics-Clinical-Data.csv
For Patrick: /content/gdrive/My Drive/Mphys project/cancerdatasem2.csv
"""
if full_dataset_choice == False:
  print("You chose to use the partial dataset")
  project_folder = "/content/gdrive/MyDrive/MPhys/Data/"
  clinical_data_filename = "COLAB-Clinical-Data.csv"
  print(os.path.join(project_folder, clinical_data_filename))
elif full_dataset_choice == True:
  print("You chose to use the full dataset")
  project_folder = "/content/gdrive/MyDrive/Data/"
  clinical_data_filename = "NSCLC-Radiomics-Clinical-Data.csv"
  print(os.path.join(project_folder, clinical_data_filename))

#===================================================================================================================
#=================== DEFINING FUNCTIONS ============================================================================
#===================================================================================================================

def open_metadata() :
    """
    Opens the metadata file using the globall defined variables 'project_folder' and 'clinical_data_filename'.

    Returns patient_IDs which will be used for checking the data is shuffled
    Returns time_markers which will be used for checking patient status at a specified timepoint
    Returns dead_statuses.

    Rory Farwell and Patrick Hastings 08/02/2022
    """
    metadata_file = os.path.join(project_folder, clinical_data_filename)
    print(f'metadata_file path: {metadata_file}')
    metadata = np.genfromtxt(metadata_file, comments = '%', dtype="str", delimiter=",")
    print(f"Length of metadata array is {len(metadata)}")

    # Retrieve data from metadata file
    patient_IDs = metadata[:,0] # selecting patient IDs from the csv file
    time_markers = metadata[:,8] # selecting the day of the last patient review from the csv file
    dead_statuses = metadata[:,9] # selecting the dead status on the last review day

    time_markers = time_markers.astype(np.float32) # converting to float
    dead_statuses = dead_statuses.astype(np.float32) # converting to float

    return patient_IDs, time_markers, dead_statuses


def patient_label(time_increment) :
    """
    Sorts patients into categories based upon when they died. 
    Category numbers up to number 9 indicate the number of half-years survived
    and one final category, category 10, is for anyone who survived for more than 5 years.

    Rory Farwell and Patrick Hastings 08/02/2022
    """
    maximum_label = math.floor(maximum_time_in_days/time_increment)
    dead_counter = 0
    alive_counter = 0
    no_info_counter = 0
    labels = np.empty(len(dead_statuses))

    for i in range(len(dead_statuses)):
      if dead_statuses[i] == 1 and time_markers[i] < maximum_time_in_days: #check patient is dead before 1800 days(5 years)
        labels[i] = math.floor(time_markers[i] / time_increment)
        print(f'Conversion: Timepoint = {time_markers[i]} was converted to Category = {labels[i]}.') #Checking the function.
      elif dead_statuses[i] == 1 and time_markers[i] > maximum_time_in_days: # add to category 10 if patient died after more than 5 years
        labels[i] = maximum_label
        print(f'Conversion: Timepoint = {time_markers[i]} was converted to Category = {labels[i]}.') #Checking the function.
      elif dead_statuses[i] == 0 and time_markers[i] > maximum_time_in_days : 
          labels[i] = maximum_label # add to category 10 if patient last seen alive after 5 years

    return labels


def training_loop():
    epoch_train_loss = 0 # will be used for plotting testing vs validation loss curves
    n_training_samples = 0
    print(f'Training for epoch {epoch+1}')
    print("=============================")

    model.train()

    for i, (images, labels) in enumerate(train_dataloader):
        images = reshape(images, (images.shape[0], 1, 264, 264, 264))
        images = images.float()

        labels = np.asarray(labels)
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)
        labels = labels.long()
        #print(f'Labels: {labels}')
        
        images = images.to(device)
        labels = labels.to(device)

        # FORWARD PASS
        outputs = model(images)
        print(f'Outputs: {outputs}')
        print(f'Labels: {labels}')
        # print (outputs)
        loss = criterion(outputs, labels)
        
        # BACKWARD PASS
        optimizer.zero_grad() #clears gradients before performing backpropagation
        loss.backward()
        optimizer.step()

        # Add the number of images in this batch to n_training_samples which will
        # be used when calculating the average loss per image in the training set
        n_training_samples += labels.shape[0]

        # Updating the total training loss of this epoch
        all_training_losses.append(loss.item())
        epoch_train_loss += loss.item()

        if (i+1)%1 == 0 :
            print(f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

    # Append the train_loss list with the total training loss for this epoch
    train_loss.append(epoch_train_loss)

    #Append the avg_train_loss list with the average training loss of this epoch
    avg_train_loss = epoch_train_loss/n_training_samples
    print(f"Average training loss list: {avg_train_loss}")

    print(f"Training loss array at end of epoch {epoch + 1}: {train_loss}. Total number of images used = {n_training_samples}.")
    print(f"Finished training for epoch {epoch + 1}")

    return avg_train_loss

def validation_loop() :
    print(f'Validation for epoch {epoch + 1}')
    print('=================================')

    model.eval()

    with torch.no_grad(): # ensuring gradients are not calculated during the validation loop
        valid_epoch_loss = 0
        n_valid_correct = 0
        n_valid_samples = 0
        for images, labels in validation_dataloader :
            images = reshape(images, (images.shape[0],1 ,264,264,264))
            images = images.float()

            labels = np.asarray(labels)
            labels = labels.astype(np.float32)
            labels = torch.from_numpy(labels)
            labels = labels.long()

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            print(f"outputs:{outputs}")

            #calculate loss of validation set
            loss = criterion(outputs, labels)
            valid_epoch_loss += loss.item()

            #max returns (value, index) 
            _,predictions = torch.max(outputs, 1)
            print(f'Network predictions = {predictions}')
            targets = labels
            print(f'Targets = {targets}')
            # print(f'predictions: {predictions}')
            # print(f'targets: {targets}')
            # print(f'correct in this batch: {(predictions == targets).sum().item()}')
            n_valid_samples += labels.shape[0]
            print(f"valid samples: {n_valid_samples}")
            n_valid_correct += (predictions == targets).sum().item()
            print(f"valid correct: {n_valid_correct}")

            for i in range(len(labels)):
              if predictions[i] == labels[i]:
                print(f"label of correctly identified patient{labels[i]}")
            #print(f'n_correct = {n_correct}. n_samples = {n_samples}')
        avg_valid_loss = valid_epoch_loss/n_valid_samples
        #valid_loss.append(valid_epoch_loss)
        acc = (100*n_valid_correct)/n_valid_samples
        print(f'Accuracy on validation set for epoch {epoch+1} = {acc:.1f}%')
        print(f'Loss on validation set = {valid_epoch_loss}')

        print(f'Finished validation for epoch {epoch+1}')
        print('=============================================')
    return avg_valid_loss

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
      images = images = reshape(images, (images.shape[0],1 ,264,264,264))
      images = images.float()

      labels = np.asarray(labels)
      labels = labels.astype(np.float32)
      labels = torch.from_numpy(labels)
      labels = labels.long()

      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      
      # max returns (value, index) 
      _,predictions = torch.max(outputs, 1)
      #_,targets = torch.max(hot_labels,1)
      targets = labels
      #print(f'predictions: {predictions}')
      #print(f'targets: {targets}')
      #n_samples += hot_labels.shape[0]
      n_samples += labels.shape[0]
      n_correct += (predictions == targets).sum().item()
      #print(f'n_correct = {n_correct}. n_samples = {n_samples}')
    
    acc = (100*n_correct)/n_samples

    return acc

def plot_loss_curves() :
  new_avg_train_loss = avg_train_loss
  new_avg_valid_loss = avg_valid_loss

  epochs = np.array(range(num_epochs)) + 1
  fig = plt.figure()
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  fig.set_size_inches(20, 10)
  plt.plot(epochs, new_avg_train_loss, label = 'Average training loss',linewidth=7.0)
  plt.plot(epochs, new_avg_valid_loss, label = 'Average validation loss',linewidth=7.0)
  plt.legend(loc='best', prop={'size': 20})
  plt.ylabel('Average Loss', fontsize = 20)
  plt.xlabel('Epoch Number', fontsize = 20)
  plt.show()
  return

def window_and_level(image, level = -600, window = 1500) :
  maxval = level + window/2
  minval = level - window/2
  wld = np.clip(image, minval, maxval)
  wld -=minval
  wld *= 1/window
  return wld


#===================================================================================================================
#=================  CLASS DEFINITIONS ==============================================================================
#===================================================================================================================

# Normalize class added 12/12/2021
# class Normalize():
#   def __init__(self):
#     pass

#   def __call__(self,vol):
#     vol =((vol-(vol.mean()))/vol.std()) + 1
#     return vol

transform = transforms.Compose(
    [transforms.ToTensor()] 
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
      image = image[mx_x+cc_shift:264+mx_x+cc_shift, mx_yz+ap_shift:264+mx_yz+ap_shift, mx_yz+lr_shift:264+mx_yz+lr_shift]

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
      self.conv1 = nn.Conv3d(1,32,2,2)
      self.pool = nn.MaxPool3d(2,2)
      self.avg_pool = nn.AvgPool3d(4)
      self.conv2 = nn.Conv3d(32,64,2,2)
      self.conv3 = nn.Conv3d(64,128,2,2)
      self.conv4 = nn.Conv3d(128,64,1,1)
      self.conv5 = nn.Conv3d(64,16,1,1)
      self.conv6 = nn.Conv3d(16,number_of_categories,1,1)

    # Defining the forward pass  (NIN method)  
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = self.avg_pool(self.conv6(x))
        x = x.view(-1,number_of_categories)
        return x
        
model = CNN().to(device) # Send the CNN to the device




#===================================================================================================================
#=================== DEFIINING VARIABLES ===========================================================================
#===================================================================================================================


# sanity check to check progress
counter = 0 


# Creating empty arrays that will be appended to later
# These will contain the patient ID and dead status (on the check day).
training_array = []
testing_array = []
validation_array = []

#===================================================================================================================
#=================== HYPER PARAMETER DEFINITION ====================================================================
#===================================================================================================================
batch_size = 4
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#number of epochs is defined at the top of the code by user input

#===================================================================================================================
#=================== MAIN CODE =====================================================================================
#===================================================================================================================
user_choice_of_time_increment = int(sys.argv[2])
number_of_categories = math.ceil(1825/user_choice_of_time_increment) + 1
print(f'Number of categories: {number_of_categories}.')
patient_labels = []
patient_IDs, time_markers, dead_statuses = open_metadata()#gathers patient data
patient_labels = patient_label(user_choice_of_time_increment)#assigns each patients a label based on survial time

patient_labels = np.reshape(patient_labels, [-1, 1])
patient_IDs = np.reshape(patient_IDs, [-1, 1])
comb_array = np.hstack((patient_IDs, patient_labels))#creates one 2d array where each row is a patient and its corresponding label
comb_array = comb_array.tolist()
print(len(patient_IDs))

category_histogram = plt.figure()
category_histogram.set_size_inches(20,10)
plt.hist(patient_labels, bins = number_of_categories)
plt.xlabel('Category', fontsize = 20)
plt.ylabel('Frequency', fontsize = 20)
plt.savefig(f'{plot_folder_path}Hist_of_Labels.png')
plt.show()



full_dataset = ImageDataset(comb_array, os.path.join(project_folder, "Textured_Masks"), transform = transform)

training_data, validation_data, test_data = torch.utils.data.random_split(full_dataset, [int(0.7*len(patient_IDs)),int(0.15*len(patient_IDs)),int(0.15*len(patient_IDs))])



train_dataloader = DataLoader(training_data, batch_size = 4, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 4, shuffle = False)
validation_dataloader = DataLoader(validation_data, batch_size = 4, shuffle = False)

summary(model, (1,264,264,264), batch_size = 4)

#============================ TRAINING AND VALIDATION LOOP =========================================================
n_total_steps = len(train_dataloader)
train_loss = []
valid_loss = []
avg_train_loss = np.empty(0)
avg_valid_loss = np.empty(0)
all_training_losses = []

for epoch in range(num_epochs):
    avg_train_loss = np.append(avg_train_loss, training_loop())
    avg_valid_loss = np.append(avg_valid_loss, validation_loop())

print('FINISHED TRAINING')
print(f'All training batch losses = {all_training_losses}')
print(f'Training losses = {train_loss}')
print(f'Average training losses = {avg_train_loss}')
print(f'Validation losses = {avg_valid_loss}')

#===================== PLOT LOSS CURVES ============================================================================
# plot_loss_curves()

new_avg_train_loss = avg_train_loss
new_avg_valid_loss = avg_valid_loss

epochs = np.array(range(num_epochs)) + 1
fig = plt.figure()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
fig.set_size_inches(20, 10)
plt.plot(epochs, new_avg_train_loss, label = 'Average training loss',linewidth=7.0)
plt.plot(epochs, new_avg_valid_loss, label = 'Average validation loss',linewidth=7.0)
plt.legend(loc='best', prop={'size': 20})
plt.ylabel('Average Loss', fontsize = 20)
plt.xlabel('Epoch Number', fontsize = 20)
plt.savefig(f'{plot_folder_path}{plot_filename}')
plt.show()
print(f'This plot has been saved at: {plot_folder_path}{plot_filename}')

#===================== TESTING LOOP ================================================================================
testing_accuracy = testing_loop()
print(f'Accuracy on testing set = {testing_accuracy:.1f}%')
